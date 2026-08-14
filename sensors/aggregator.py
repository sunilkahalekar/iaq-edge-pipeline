"""
aggregator.py
=============
Polls all four sensors, each at its own natural cadence, and emits ONE
aggregated row every 60 seconds aligned to the wall-clock minute boundary,
using naive LOCAL time (datetime.now(), no tzinfo) -- deliberately matching
iaq_pipeline_pi.py's WindowBuffer, which stamps window_start from a plain
datetime.now() call. Using a different clock convention here (e.g. UTC)
would silently break every downstream join unless the Pi happens to run
UTC as its local zone.

IMPORTANT -- this does NOT mean sensor_readings.window_start will exact-
string-match ct_vectors.window_start. ct_vectors' window_start is NOT
minute-boundary-aligned at all: WindowBuffer stamps it from whatever
timestamp the first tick of a fresh window happened to land on (see
iaq_pipeline_pi.py's WindowBuffer.push()), which drifts based on Pi boot
time and tick timing, not wall-clock minute marks. sensor_readings' rows
ARE minute-aligned (by construction here). The two tables must be joined
by flooring BOTH sides to the minute (Python: dt.replace(second=0,
microsecond=0), same "floor_to_minute" approach
merge_vision_and_sensor_data.py already uses for exactly this reason) --
see iaq_forecast.py, which does this at query time. Do not rely on exact
window_start string equality between these two tables.

DESIGN, MIRRORING iaq_pipeline_pi.py's OWN RULES:
  - Each sensor's serial/I2C/GPIO handle is owned by exactly ONE thread,
    never shared -- same reason db_writer.py gives for its single-writer
    design (these interfaces are not thread-safe). One polling thread per
    sensor, each writing only into its own small ring buffer.
  - The 60s aggregation just reads the latest buffered values -- it never
    touches a sensor handle directly, so it can't contend with a polling
    thread mid-transaction.
  - A sensor that's failing (all reads returning None) does NOT stop the
    other three from reporting -- each column in the aggregated row is
    independently null-able. This matches how the training pipeline
    already handles missing columns (ffill/bfill/fillna(0.0) in
    feature_engineering.py's build_base_columns()) -- a temporarily dead
    sensor degrades gracefully instead of crashing the whole pipeline.

CADENCE PER SENSOR (why these specific numbers):
  - PMS5003: streams ~1/sec on its own; poll as fast as frames arrive.
  - MH-Z19B: command/response, no benefit to polling faster than ~2s;
    the sensor's internal NDIR measurement cycle is a few seconds anyway.
  - SGP40: MUST be sampled at ~1Hz for the gas-index algorithm's running
    baseline to converge (see sgp40.py's docstring) -- sampled every
    ~1.0s regardless of the 60s output cadence.
  - DHT22: datasheet minimum 2s between reads; sampled every 2.5s.

OUTPUT COLUMN NAMES: deliberately match what
context-aware-bilstm/src/modeling/feature_engineering.py's
build_base_columns() expects verbatim (temp, hum, pm1, pm2_5, pm10, co2,
voc) -- this is the whole point of the aggregator, producing a row that
slots into the existing feature pipeline with zero renaming.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from datetime import datetime, timedelta


class _SensorPoller:
    """Owns one sensor's handle in one dedicated thread. Buffers the last
    `buffer_seconds` worth of successful reads; read() returns None entries
    are simply not buffered (they don't corrupt the window mean)."""

    def __init__(self, name, factory, poll_interval, buffer_seconds=90):
        self._name = name
        self._factory = factory
        self._poll_interval = poll_interval
        self._buffer = deque()
        self._buffer_seconds = buffer_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sensor = None
        self._last_error_log = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"poll-{name}")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self):
        try:
            self._sensor = self._factory()
        except Exception as exc:  # noqa: BLE001 -- must not crash the process
            print(f"[{self._name}] failed to initialize: {exc!r} -- this sensor "
                  f"will report no data until the process is restarted.")
            return

        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                reading = self._sensor.read()
            except Exception as exc:  # noqa: BLE001
                reading = None
                now = time.monotonic()
                if now - self._last_error_log > 30:
                    print(f"[{self._name}] read error: {exc!r}")
                    self._last_error_log = now
            if reading is not None:
                with self._lock:
                    self._buffer.append((time.time(), reading))
                    cutoff = time.time() - self._buffer_seconds
                    while self._buffer and self._buffer[0][0] < cutoff:
                        self._buffer.popleft()
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, self._poll_interval - elapsed))

    def latest(self):
        with self._lock:
            return self._buffer[-1][1] if self._buffer else None

    def window_values(self, since_ts: float):
        with self._lock:
            return [r for ts, r in self._buffer if ts >= since_ts]


def _mean_or_none(values):
    return statistics.fmean(values) if values else None


class SensorAggregator:
    """
    Usage:
        agg = SensorAggregator(pms_port="/dev/ttyUSB0", mhz_port="/dev/ttyUSB1")
        agg.start()
        for row in agg.iter_windows():   # blocks, yields one dict per minute
            db_writer.write("sensor_readings", SENSOR_COLUMNS, row)
    """

    def __init__(self, pms_port="/dev/ttyUSB0", mhz_port="/dev/ttyUSB1",
                 i2c_bus=1, dht_gpio_pin=4, window_seconds=60):
        self._window_seconds = window_seconds
        self._pollers = []

        def _mk_pms():
            from .pms5003 import PMS5003Reader
            return PMS5003Reader(port=pms_port)

        def _mk_mhz():
            from .mhz19b import MHZ19BReader
            return MHZ19BReader(port=mhz_port)

        def _mk_dht():
            from .dht22 import DHT22Reader
            return DHT22Reader(gpio_pin=dht_gpio_pin)

        self._dht_poller = _SensorPoller("dht22", _mk_dht, poll_interval=2.5)
        self._pms_poller = _SensorPoller("pms5003", _mk_pms, poll_interval=1.0)
        self._mhz_poller = _SensorPoller("mhz19b", _mk_mhz, poll_interval=2.0)

        # SGP40 needs the DHT22's latest temp/hum for compensation, and
        # must itself be sampled at ~1Hz -- given its own dedicated poller
        # whose factory closes over the DHT poller to read compensation
        # values without touching DHT's serial/GPIO handle directly.
        def _mk_sgp():
            from .sgp40 import SGP40Reader
            base = SGP40Reader(bus=i2c_bus)
            dht_poller = self._dht_poller

            class _CompensatedSGP40:
                def read(self):
                    latest_dht = dht_poller.latest()
                    t = latest_dht.temp_c if latest_dht else 25.0
                    h = latest_dht.humidity_pct if latest_dht else 50.0
                    return base.read(temp_c=t, humidity_pct=h)

            return _CompensatedSGP40()

        self._sgp_poller = _SensorPoller("sgp40", _mk_sgp, poll_interval=1.0)

        self._pollers = [self._dht_poller, self._pms_poller,
                          self._mhz_poller, self._sgp_poller]

    def start(self):
        for p in self._pollers:
            p.start()

    def stop(self):
        for p in self._pollers:
            p.stop()

    def _next_minute_boundary(self) -> float:
        # Naive local time, deliberately -- matches iaq_pipeline_pi.py's
        # WindowBuffer (datetime.now(), no tzinfo). .timestamp() on a naive
        # datetime is interpreted as local time by Python and converted to
        # a true Unix epoch float, so this stays consistent with the
        # epoch-based time.time() values used elsewhere in this file.
        now = datetime.now()
        nxt = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
        return nxt.timestamp()

    def _build_window_row(self, window_start_ts: float, window_end_ts: float) -> dict:
        pm_readings = self._pms_poller.window_values(window_start_ts)
        mhz_readings = self._mhz_poller.window_values(window_start_ts)
        sgp_readings = self._sgp_poller.window_values(window_start_ts)
        dht_readings = self._dht_poller.window_values(window_start_ts)

        window_start_str = datetime.fromtimestamp(window_start_ts).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        return {
            "window_start": window_start_str,
            "pm1": _mean_or_none([r.pm1_0 for r in pm_readings]),
            "pm2_5": _mean_or_none([r.pm2_5 for r in pm_readings]),
            "pm10": _mean_or_none([r.pm10 for r in pm_readings]),
            "co2": _mean_or_none([r.co2_ppm for r in mhz_readings]),
            # last value, not mean: the gas-index algorithm's own internal
            # state already smooths/converges the index over time -- taking
            # a mean of already-converged index values is redundant and
            # can lag a genuine step change.
            "voc": sgp_readings[-1].voc_index if sgp_readings else None,
            "temp": _mean_or_none([r.temp_c for r in dht_readings]),
            "hum": _mean_or_none([r.humidity_pct for r in dht_readings]),
        }

    def iter_windows(self):
        """Blocks until each wall-clock minute boundary, then yields the
        aggregated row for the window that just closed. Runs forever."""
        next_boundary = self._next_minute_boundary()
        while True:
            sleep_for = next_boundary - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            window_start = next_boundary - self._window_seconds
            row = self._build_window_row(window_start, next_boundary)
            yield row
            next_boundary += self._window_seconds
