"""
dht22.py
========
Driver for the AM2302/DHT22 temperature+humidity sensor over a single GPIO
data pin (1-wire, bit-banged serial protocol).

WHY THIS WRAPS adafruit-circuitpython-dht INSTEAD OF BIT-BANGING GPIO
DIRECTLY IN PYTHON:
  The DHT22 protocol communicates bit values by pulse duration (~26-28us
  for a 0, ~70us for a 1) that pure-Python GPIO polling on a general-
  purpose, non-realtime Linux kernel cannot reliably resolve -- a
  scheduler hiccup of a few hundred microseconds corrupts the read. This
  is why hand-rolled pure-Python DHT22 libraries have a well-earned
  reputation for high failure rates. adafruit-circuitpython-dht wraps a
  small C extension that does the timing-critical bit-banging, which is
  the standard, correct way to read this sensor from a Pi running
  Raspberry Pi OS. Install via:
      pip install adafruit-circuitpython-dht
  (also needs `sudo apt install libgpiod2` on Bookworm).

KNOWN CHARACTERISTIC, NOT A BUG: even with the C-backed library, DHT22
reads fail checksum some fraction of the time (community reports vary,
commonly 5-20%) -- this is normal for the sensor/protocol, not something
a "correct" driver eliminates. This module retries a bounded number of
times and returns None if all retries fail; the caller must handle None
(skip this field for the current window) rather than treat a single
failed read as fatal.

SAMPLING RATE: the datasheet specifies a minimum 2-second interval
between reads. Do not call read() faster than that -- the aggregator
enforces this, not this module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class DHT22Reading:
    temp_c: float
    humidity_pct: float
    timestamp: float


class DHT22Reader:
    def __init__(self, gpio_pin: int = 4, retries: int = 3, retry_delay: float = 2.1):
        import adafruit_dht
        import board

        pin = getattr(board, f"D{gpio_pin}")
        self._sensor = adafruit_dht.DHT22(pin, use_pulseio=False)
        self._retries = retries
        self._retry_delay = retry_delay

    def close(self):
        self._sensor.exit()

    def read(self) -> DHT22Reading | None:
        """Retries up to self._retries times (2+ second spacing per the
        datasheet minimum). Returns None if every attempt fails -- this is
        expected to happen occasionally, not exceptional."""
        for attempt in range(self._retries):
            try:
                temp_c = self._sensor.temperature
                humidity = self._sensor.humidity
                if temp_c is not None and humidity is not None:
                    return DHT22Reading(
                        temp_c=float(temp_c), humidity_pct=float(humidity),
                        timestamp=time.time(),
                    )
            except RuntimeError:
                pass  # expected occasional checksum/timing failure -- retry
            if attempt < self._retries - 1:
                time.sleep(self._retry_delay)
        return None


if __name__ == "__main__":
    r = DHT22Reader()
    for _ in range(10):
        print(r.read())
        time.sleep(2.5)
    r.close()
