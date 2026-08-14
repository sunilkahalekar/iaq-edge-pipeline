"""
pms5003.py
==========
Driver for the Plantower PMS5003 laser particulate sensor (PM1.0/PM2.5/PM10)
over UART, 9600 baud 8N1.

WIRING / PI CONFIG NOTE (read before deploying):
  The Pi 4's primary hardware UART (PL011, /dev/ttyAMA0) is routed to
  Bluetooth by default; the GPIO14/15 pins get the "mini-UART" instead,
  which is not clock-stable enough for reliable sensor UART at 9600 baud
  when the CPU frequency scales. To use GPIO14/15 for this sensor
  reliably, disable Bluetooth in /boot/firmware/config.txt:
      dtoverlay=disable-bt
  and reboot. Alternatively (simpler, no config.txt changes, what this
  driver was written assuming by default), use a cheap USB-to-TTL adapter
  (CP2102/CH340) and point PORT at /dev/ttyUSB0 or similar -- this avoids
  UART pin contention entirely if MH-Z19B is also using serial (see
  mhz19b.py's own wiring note; you can't put both sensors on the same
  single hardware UART).

PROTOCOL (Plantower PMS5003 datasheet):
  In its default "active mode", the sensor streams one 32-byte frame
  approximately once per second without needing any command to be sent:

    byte 0-1   : start bytes 0x42 0x4D
    byte 2-3   : frame length (following byte count) -- always 28 (0x001C)
    byte 4-5   : PM1.0  (CF=1, "standard particle" calibration)
    byte 6-7   : PM2.5  (CF=1)
    byte 8-9   : PM10   (CF=1)
    byte 10-11 : PM1.0  (atmospheric environment)
    byte 12-13 : PM2.5  (atmospheric environment)
    byte 14-15 : PM10   (atmospheric environment)
    byte 16-25 : particle counts per size bin (0.3/0.5/1.0/2.5/5.0/10um)
    byte 26-27 : reserved
    byte 28-29 : reserved
    byte 30-31 : checksum = sum(byte 0..29), big-endian

  This driver reads the "atmospheric environment" fields (bytes 10-15),
  not the CF=1 "standard particle" fields -- the atmospheric fields are
  the ambient-air concentration values; CF=1 is a factory calibration
  reference and is NOT what you want for real air-quality readings. This
  is a common mistake in PMS5003 example code found online -- confirm
  against the datasheet before changing which field this reads.

WARM-UP: after power-on (or waking from sleep via the SET pin, not used
by this driver), the fan needs ~30 seconds to reach a stable reading.
The first few reads after connecting will be present but unreliable --
the aggregator should discard the first ~30s of readings after startup,
not this driver's job to know about aggregation-level warm-up policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import serial

START_1 = 0x42
START_2 = 0x4D
FRAME_LEN = 32  # 2 start + 2 length + 26 data + 2 checksum... total bytes on wire


@dataclass
class PMS5003Reading:
    pm1_0: float
    pm2_5: float
    pm10: float
    timestamp: float


class PMS5003Reader:
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 9600,
                 timeout: float = 2.0):
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)

    def close(self):
        self._ser.close()

    def _read_frame(self) -> bytes | None:
        """Scans for the 0x42 0x4D start sequence and reads one full frame.
        Returns None on timeout or checksum failure (caller should retry)."""
        deadline = time.monotonic() + self._ser.timeout * 4
        while time.monotonic() < deadline:
            b = self._ser.read(1)
            if not b or b[0] != START_1:
                continue
            b2 = self._ser.read(1)
            if not b2 or b2[0] != START_2:
                continue
            rest = self._ser.read(FRAME_LEN - 2)
            if len(rest) != FRAME_LEN - 2:
                return None  # short read -- serial timeout mid-frame
            frame = bytes([START_1, START_2]) + rest
            checksum_calc = sum(frame[:30]) & 0xFFFF
            checksum_recv = (frame[30] << 8) | frame[31]
            if checksum_calc != checksum_recv:
                continue  # corrupt frame, keep scanning for next start seq
            return frame
        return None

    def read(self) -> PMS5003Reading | None:
        """Blocks up to ~4x the serial timeout waiting for one valid frame.
        Returns None if no valid frame arrived -- caller decides whether to
        retry or skip this cycle. Never raises on malformed/corrupt data."""
        frame = self._read_frame()
        if frame is None:
            return None
        pm1_0 = (frame[10] << 8) | frame[11]
        pm2_5 = (frame[12] << 8) | frame[13]
        pm10 = (frame[14] << 8) | frame[15]
        return PMS5003Reading(
            pm1_0=float(pm1_0), pm2_5=float(pm2_5), pm10=float(pm10),
            timestamp=time.time(),
        )


if __name__ == "__main__":
    import sys
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
    r = PMS5003Reader(port=port)
    print(f"Reading from {port} -- warming up 30s (fan spin-up)...")
    time.sleep(30)
    for _ in range(10):
        reading = r.read()
        print(reading)
        time.sleep(1)
    r.close()
