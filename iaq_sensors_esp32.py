"""
IAQ Sensor Ingestion — ESP32 relay variant
=============================================
Alternative to iaq_sensors.py (which talks to sensors directly via
pyserial/smbus2/GPIO on the Pi). This version instead reads one JSON line
per minute from an ESP32 (esp32_sensor_node/esp32_sensor_node.ino) over a
USB-serial connection, and writes it straight into the same
sensor_readings table. Pick ONE of the two ingestion scripts per
deployment — don't run both, they'd both write into the same table.

Runs as its own process (own systemd unit, iaq-sensors-esp32.service),
same reasoning as iaq_sensors.py: independent of the camera pipeline, own
AsyncDBWriter, safe to crash/restart without affecting anything else.

WHY A SEPARATE SCRIPT INSTEAD OF BRANCHING INSIDE iaq_sensors.py: the two
ingestion paths have almost nothing in common at the implementation level
-- one does sensor-protocol parsing directly, this one does line-based
JSON parsing off a serial port and basic schema validation. Keeping them
as separate files means neither has dead code paths for the other, and
the systemd units can be swapped without touching Python source.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime

from db_writer import make_sensor_writer

EXPECTED_FIELDS = ["pm1", "pm2_5", "pm10", "co2", "voc", "temp", "hum"]

_STOP = False


def _handle_sigterm(signum, frame):
    global _STOP
    _STOP = True


def parse_esp32_line(line: str) -> dict | None:
    """Parses one JSON line from the ESP32 into a sensor_readings row.
    Returns None (and logs why) on any malformed input -- a corrupt or
    partial line from a USB hiccup must not crash the ingestion loop.
    Adds window_start as the current wall-clock minute (naive local time,
    matching every other timestamp in this DB -- see CLAUDE.md)."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        print(f"WARNING: malformed JSON from ESP32, dropping line: {line!r} ({exc})")
        return None

    if not isinstance(data, dict):
        print(f"WARNING: expected a JSON object from ESP32, got: {line!r}")
        return None

    row = {"window_start": datetime.now().replace(second=0, microsecond=0)
           .strftime("%Y-%m-%d %H:%M:%S")}
    missing = []
    for field in EXPECTED_FIELDS:
        value = data.get(field)
        if value is None:
            missing.append(field)
        row[field] = value

    if missing:
        print(f"[{row['window_start']}] ESP32 reported no data for: {missing}")

    return row


def run(port: str, baud: int, db_path: str):
    import serial   # imported here, not at module level, so parse_esp32_line()
                     # and other logic can be tested/imported without pyserial
                     # installed -- only run() (the actual serial I/O) needs it

    sensor_writer, shared = make_sensor_writer(db_path)

    ser = None
    try:
        while not _STOP:
            if ser is None:
                try:
                    ser = serial.Serial(port, baudrate=baud, timeout=90)
                    print(f"Connected to ESP32 on {port} @ {baud} baud.")
                except serial.SerialException as exc:
                    print(f"Could not open {port}: {exc!r} -- retrying in 10s.")
                    time.sleep(10)
                    continue

            try:
                raw = ser.readline()
            except serial.SerialException as exc:
                print(f"Serial read error: {exc!r} -- reconnecting.")
                ser.close()
                ser = None
                continue

            if not raw:
                # timeout with no data -- ESP32 emits once/minute, a 90s
                # read timeout with nothing is a real problem (device
                # unplugged, firmware hung), not a transient blip
                print("WARNING: no data from ESP32 in 90s -- check connection.")
                continue

            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            row = parse_esp32_line(line)
            if row is not None:
                sensor_writer.write(row)
                present = {k: v for k, v in row.items() if k != "window_start" and v is not None}
                print(f"[{row['window_start']}] wrote sensor_readings row: {present}")
    finally:
        if ser is not None:
            ser.close()
        shared.close()
        print("iaq_sensors_esp32: stopped cleanly.")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyUSB0",
                     help="Serial device the ESP32 enumerates as -- confirm with "
                          "`ls /dev/ttyUSB* /dev/ttyACM*` after plugging in.")
    ap.add_argument("--baud", type=int, default=115200,
                     help="Must match Serial.begin() in the ESP32 firmware.")
    ap.add_argument("--db-path", default="/home/pi4/iaq_data/iaq.db")
    return ap.parse_args()


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)
    run(args.port, args.baud, args.db_path)


if __name__ == "__main__":
    sys.exit(main() or 0)
