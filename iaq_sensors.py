"""
IAQ Sensor Ingestion — Raspberry Pi 4 Edition
===============================================
Runs as its OWN process (own systemd unit, iaq-sensors.service) alongside
iaq_pipeline_pi.py's camera pipeline -- not a thread inside it. The two
are functionally independent (sensors don't touch the camera; the camera
doesn't touch I2C/UART) and each already has a bounded, well-behaved
single-writer-thread relationship with the shared SQLite file, so running
them as separate processes is safe -- see db_writer.py's
make_sensor_writer() docstring for why this doesn't reintroduce the
"database is locked" bug the camera pipeline already fixed once.

Reads four pollutant/environmental sensors (PMS5003, MH-Z19B, SGP40,
DHT22 -- see sensors/ for wiring notes and protocol details per sensor)
and writes one aggregated row to the sensor_readings table every 60
seconds, aligned to the wall-clock minute boundary to match
ct_vectors.window_start's existing convention.
"""

import signal
import sys

from db_writer import make_sensor_writer
from sensors.aggregator import SensorAggregator

# ══════════════════════════════ CONFIGURATION ══════════════════════════════

# PI: adjust to match your actual wiring. PMS5003 and MH-Z19B both need a
#     UART; the Pi 4 has exactly one hardware UART exposed on GPIO14/15
#     (shared with Bluetooth by default -- see sensors/pms5003.py's
#     wiring note). Simplest reliable setup: one sensor on the GPIO UART
#     (after disabling Bluetooth), the other on a cheap USB-to-TTL
#     adapter. Confirm actual device paths with `ls /dev/ttyUSB* /dev/ttyAMA*`
#     after plugging in -- USB enumeration order is not guaranteed stable
#     across reboots if you have other USB-serial devices attached; if you
#     need a stable path, use /dev/serial/by-id/... instead.
PMS5003_PORT = "/dev/ttyUSB0"
MHZ19B_PORT = "/dev/ttyUSB1"

# PI: I2C bus for SGP40. Bus 1 is the standard GPIO I2C on all Pi 4 models.
#     Enable I2C first: `sudo raspi-config` -> Interface Options -> I2C.
I2C_BUS = 1

# PI: BCM GPIO pin number for DHT22's data line (not the physical pin
#     number -- BCM4 is physical pin 7). Whatever pin you choose here must
#     match your wiring; there's no way to auto-detect this.
DHT22_GPIO_PIN = 4

DB_PATH = "/home/pi4/iaq_data/iaq.db"
DB_COMMIT_INTERVAL_SECONDS = 10.0  # sensor writes are already only 1/min;
                                    # no need for the camera pipeline's
                                    # tighter 2s commit interval here.

_STOP = False


def _handle_sigterm(signum, frame):
    global _STOP
    _STOP = True


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    sensor_writer, shared = make_sensor_writer(
        DB_PATH, commit_every_sec=DB_COMMIT_INTERVAL_SECONDS
    )

    agg = SensorAggregator(
        pms_port=PMS5003_PORT, mhz_port=MHZ19B_PORT,
        i2c_bus=I2C_BUS, dht_gpio_pin=DHT22_GPIO_PIN,
        window_seconds=60,
    )
    agg.start()
    print(f"iaq_sensors: polling started, writing to {DB_PATH} every 60s. "
          f"Ctrl-C or SIGTERM to stop.")

    try:
        for row in agg.iter_windows():
            if _STOP:
                break
            missing = [k for k, v in row.items() if v is None and k != "window_start"]
            if missing:
                print(f"[{row['window_start']}] missing this window: {missing}")
            sensor_writer.write(row)
    finally:
        agg.stop()
        shared.close()
        print("iaq_sensors: stopped cleanly.")


if __name__ == "__main__":
    sys.exit(main() or 0)
