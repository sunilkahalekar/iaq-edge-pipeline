"""
mhz19b.py
=========
Driver for the Winsen MH-Z19B NDIR CO2 sensor over UART, 9600 baud 8N1,
using its command/response protocol (not the PWM output pin -- PWM is an
alternative wiring this driver does not support).

WIRING / PI CONFIG NOTE:
  If PMS5003 is already using the Pi's one hardware UART, this sensor
  needs a SEPARATE serial port -- typically a second USB-to-TTL adapter
  (/dev/ttyUSB1 or similar). Do not try to share one UART between two
  sensors. See pms5003.py's wiring note for the Bluetooth/mini-UART
  caveat if you instead plan to use the GPIO14/15 hardware UART for one
  of the two sensors.

PROTOCOL (Winsen MH-Z19B datasheet, "Read CO2 concentration" command):
  Request  (9 bytes): FF 01 86 00 00 00 00 00 79
    byte 0    : 0xFF (start)
    byte 1    : 0x01 (sensor "number", always 1 for single-sensor UART)
    byte 2    : 0x86 (command: read CO2 concentration)
    byte 3-7  : 0x00 (unused for this command)
    byte 8    : checksum

  Response (9 bytes): FF 86 [CO2_HB] [CO2_LB] [T] [S] [U] [V] [checksum]
    byte 0    : 0xFF
    byte 1    : 0x86 (echoes the command)
    byte 2-3  : CO2 ppm, big-endian (byte2 * 256 + byte3)
    byte 4-7  : sensor-internal status/temperature bytes, not parsed here
    byte 8    : checksum

  Checksum (both directions): checksum = (0xFF - (sum(bytes[1:8]) & 0xFF) + 1) & 0xFF

WARM-UP: MH-Z19B needs a 3-minute preheat after power-on before readings
are trustworthy (self-heating NDIR source stabilizing), and its factory
auto-baseline-calibration (ABC) assumes the sensor sees fresh outdoor-ish
air (~400ppm) at least once every 24h -- if this Pi's camera room is never
unoccupied/well-ventilated, ABC can drift the baseline. Consider disabling
ABC (command 0x79) and doing periodic manual calibration instead if the
room is continuously occupied; that's a deployment decision, not something
this driver decides for you.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import serial

READ_CO2_CMD = bytes([0xFF, 0x01, 0x86, 0x00, 0x00, 0x00, 0x00, 0x00, 0x79])


@dataclass
class MHZ19BReading:
    co2_ppm: float
    timestamp: float


def _checksum(packet: bytes) -> int:
    s = sum(packet[1:8]) & 0xFF
    return (0xFF - s + 1) & 0xFF


class MHZ19BReader:
    def __init__(self, port: str = "/dev/ttyUSB1", baudrate: int = 9600,
                 timeout: float = 2.0):
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)

    def close(self):
        self._ser.close()

    def read(self) -> MHZ19BReading | None:
        """Sends the read-CO2 command and parses the 9-byte response.
        Returns None on timeout/checksum mismatch -- never raises."""
        self._ser.reset_input_buffer()
        self._ser.write(READ_CO2_CMD)
        resp = self._ser.read(9)
        if len(resp) != 9:
            return None
        if resp[0] != 0xFF or resp[1] != 0x86:
            return None
        if _checksum(resp) != resp[8]:
            return None
        co2 = (resp[2] << 8) | resp[3]
        return MHZ19BReading(co2_ppm=float(co2), timestamp=time.time())


if __name__ == "__main__":
    import sys
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB1"
    r = MHZ19BReader(port=port)
    for _ in range(10):
        print(r.read())
        time.sleep(2)
    r.close()
