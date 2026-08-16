# CLAUDE.md — internals reference for iaq-edge-pipeline

Read this before making cross-cutting changes (touching the DB schema,
adding a fourth process, changing timestamp handling). For how to *run*
this repo, see [README.md](README.md) first — this file is for anyone
about to modify the code.

## How this repo came together

Started as a single-process camera pipeline (`iaq_pipeline_pi.py` +
`db_writer.py` + `dashboard.py`). Two more independent processes were
added later, each writing to the same SQLite file: `iaq_sensors.py`
(pollutant sensors → `sensor_readings`) and, in the companion repo
[`context-aware-bilstm-edge`](https://github.com/sunilkahalekar/context-aware-bilstm-edge),
`iaq_forecast.py` (BiLSTM inference → `forecasts`, reading this repo's
`ct_vectors` + `sensor_readings` live). `iaq_forecast.py` isn't vendored
into this repo — it's deployed by copying it (and its own
`feature_engineering.py`/`models.py`) into this repo's directory on the
Pi at deploy time, the same way `export_model.py` already treats the YOLO
model as something built elsewhere and copied in, not built here.

## The three-process architecture, and why it's three processes, not one

```
iaq_pipeline_pi.py (camera, 1Hz)  ──┐
iaq_sensors.py (pollutants, ~1/min) ├──► ONE SQLite file, one AsyncDBWriter EACH
iaq_forecast.py (BiLSTM, 1/min)     ┘        (three independent connections)
                                              │
                                    dashboard.py (read-only)
```

Each process owns exactly one `AsyncDBWriter` (one thread, one
connection) — never share a connection across threads, and never give one
process more than one writer connection. This is NOT the same failure
mode as the original "database is locked" bug (see `AsyncDBWriter`'s own
docstring in `db_writer.py`): that bug came from *one process* opening a
separate connection *per table*, each independently holding a long-lived
transaction. Three *different processes*, each with exactly one
connection and short, deadline-driven commits, is safe under SQLite's WAL
mode — they'll occasionally, briefly serialize on the write lock (covered
by the 30s busy timeout), which is a non-issue at these write rates (1Hz
camera writes are batched every ~2s; sensor and forecast writes happen
once a minute).

**Why not one process for everything?** Camera inference is
latency-sensitive (a missed 1Hz tick is gone forever); sensor polling and
forecasting are not. Coupling them means a slow ONNX inference call could
stall a camera frame, or a wedged serial port could stall the pipeline
that owns the camera. Independent processes with independent `Restart=
on-failure` systemd units mean any one of the three can crash and restart
without taking the others down.

## Timestamp handling — the part most likely to bite you

**Clock convention: naive local time everywhere, never UTC.**
`iaq_pipeline_pi.py`'s `WindowBuffer` stamps `ct_vectors.window_start`
from a plain `datetime.now()` (no tzinfo). `sensors/aggregator.py` was
originally written using `datetime.now(timezone.utc)` — a real bug, fixed
during the sensor-ingestion work, because on any Pi not explicitly
configured for UTC, the two tables' timestamps would never correspond to
the same real-world minute. If you add a fourth timestamp-writing
process, use naive `datetime.now()`, not UTC, and say so in a comment —
this is exactly the kind of thing that fails silently rather than loudly.

**`ct_vectors.window_start` is NOT minute-aligned.** `WindowBuffer.push()`
stamps `window_start_ts` from whichever timestamp happens to be the first
tick of a fresh 60-tick window — that drifts based on Pi boot time and
tick timing, not wall-clock `:00` boundaries. `sensor_readings.window_start`
**is** minute-aligned (`sensors/aggregator.py` deliberately snaps to the
next minute boundary). Never join these two tables on exact
`window_start` string equality — floor both to the minute first
(`dt.replace(second=0, microsecond=0)` in Python, or the SQL equivalent).
This is the same "floor_to_minute" approach
`context-aware-bilstm/src/data_pipeline/merge_vision_and_sensor_data.py`
already uses for the same reason, offline. `iaq_forecast.py` (in the edge
repo) does this at query time — don't add a second join path that skips it.

## Column name parity with feature_engineering.py is deliberate, not incidental

`db_writer.py`'s `CT_COLUMNS` and `SENSOR_COLUMNS` match
`context-aware-bilstm/src/modeling/feature_engineering.py`'s
`build_base_columns()` expected names **verbatim** — `temp`, `hum`,
`pm1`, `pm2_5`, `pm10`, `co2`, `voc`, `M{n}_tau_open`, `M{n}_f_trans`,
`M{n}_rho_open`, `M{n}_eps_max`, `M{n}_phi_open`, `n_person`,
`mu_motion`, `sigma2_motion`. This means a query against this DB can be
fed straight into that module's feature-engineering functions with zero
renaming/mapping layer. If you ever add a sensor or a vision feature,
name its column to match what the modeling stage would call it, not
what's convenient for the sensor driver — a renaming layer is exactly
the kind of thing that silently drifts out of sync.

## The door-orientation bug the edge work found here

Not a bug in *this* repo's code, but load-bearing for anything reading
`ct_vectors` for inference: `feature_engineering.py`'s "FIX 15" door
reversal (`hi + lo - value`) needs `hi`/`lo` computed from the *training*
data's full history, not recomputed from whatever small live window is
being queried. Proven wrong on real data (a 30-row slice gave `[3.0,
3.0]` instead of the correct `[0.0, 3.0]`). If you write a second
consumer of `ct_vectors`/`sensor_readings` beyond `iaq_forecast.py`,
reuse `feature_engineering.load_bundle()`'s persisted `door_lo`/`door_hi`
— don't let `apply_door_orientation_fix()` recompute them from your query
result.

## Two sensor-integration paths — pick one, know why the other exists

`iaq_sensors.py` (+ `sensors/*.py`) wires all four sensors straight to
the Pi. `iaq_sensors_esp32.py` (+ `esp32_sensor_node/`) instead reads one
JSON line/minute from an ESP32 that owns the sensors itself. The reason
for the second path: the Pi 4 has exactly one usable hardware UART, and
two of the four sensors (PMS5003, MH-Z19B) each need their own UART — the
direct-wired path works around this with `dtoverlay=disable-bt` + a
USB-to-TTL adapter, which is a real but slightly awkward constraint; the
ESP32 (three independent hardware UARTs) sidesteps it entirely. Both
paths write into the identical `sensor_readings` schema — **never run
both at once**, they'd race on the same table. If you add a fifth
sensor, decide which path owns it and update only that path's driver
code; don't half-implement it in both.

**ESP32 boot-log lines leak onto the same serial connection as the JSON
data.** Every ESP32 power-on/reset prints diagnostic text (`ets Jul 29
2019 12:21:46`, `rst:0x1 (POWERON_RESET)...`) over UART0 — the same
connection `iaq_sensors_esp32.py` reads the JSON stream from. This is
normal, not a firmware bug; `parse_esp32_line()` must treat
non-JSON/malformed lines as "drop and continue," never as a fatal error
— confirmed with a direct test simulating this exact scenario during the
ESP32 integration work.

## Sensor hardware gotchas (see `sensors/*.py` docstrings for full detail)

- **PMS5003**: ~30s fan warm-up needed after power-on before readings are
  trustworthy. Reads "atmospheric environment" fields (bytes 10-15), not
  the "CF=1 standard particle" fields (bytes 4-9) — a common mistake in
  example code found online.
- **MH-Z19B**: 3-minute preheat. Auto-baseline-calibration assumes fresh
  ~400ppm air at least once every 24h; a continuously-occupied room can
  drift the baseline — disable ABC and calibrate manually if that's your
  deployment.
- **SGP40**: the "VOC Index" is a *stateful* running-baseline algorithm
  (Sensirion's `sensirion-gas-index-algorithm`, not hand-rolled — their
  algorithm internals aren't published for reimplementation) that needs
  continuous ~1Hz sampling to converge. `sensors/aggregator.py` samples it
  at 1Hz internally regardless of the 60s output cadence — don't reduce
  that internal rate to "optimize" polling.
- **DHT22**: checksum failures some fraction of the time (community
  reports vary, commonly 5-20%) are normal for this sensor/protocol, not
  a driver bug. `sensors/dht22.py` retries and returns `None` on
  exhaustion; callers must treat `None` as "no data this cycle," not an
  error.
- **Two UART sensors, one hardware UART**: the Pi 4 has exactly one
  full-featured hardware UART on GPIO14/15, shared with Bluetooth by
  default. PMS5003 and MH-Z19B can't both use it — free it via
  `dtoverlay=disable-bt` for one, and use a USB-to-TTL adapter for the
  other.

## Process priority (systemd `Nice=`)

`iaq-pipeline.service` (camera): `Nice=0`. `iaq-sensors.service`: `Nice=5`.
`iaq-forecast.service` (in the edge repo): `Nice=10`. Lower Nice = higher
scheduling priority. This ordering is deliberate: a missed camera frame
is unrecoverable, a late sensor sample is a minor gap, a late forecast is
barely noticeable. Don't rebalance this without the same reasoning.

## Known open issues

- **Sensor ingestion has zero built-in redundancy.** If a serial port or
  I2C bus becomes unavailable mid-run, that sensor's columns go
  permanently `NULL` until the service restarts. No auto-reconnect logic
  exists yet.
- **Nothing in this repo has been run against real sensor hardware or a
  physical Pi as of the sensor-ingestion work** — the protocol/checksum
  math (PMS5003, MH-Z19B, SGP40's CRC-8) was verified against known-good
  datasheet/Sensirion test vectors in isolation, and the DB/aggregation
  logic was integration-tested against a realistic synthetic database,
  but end-to-end hardware behavior is unconfirmed.
- **Single-camera design** — one camera frame drives door state for all
  configured machines and the overall person count (pre-existing, not
  part of the sensor/forecast work).
