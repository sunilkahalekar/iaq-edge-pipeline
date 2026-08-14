"""
sgp40.py
========
Driver for the Sensirion SGP40 VOC sensor over I2C (address 0x59).

WIRING: standard I2C -- SDA/SCL on the Pi's I2C bus (usually bus 1,
GPIO2/3). Enable I2C via `raspi-config` first. No pin contention with the
UART sensors (pms5003.py / mhz19b.py) since this is a different bus.

PROTOCOL (Sensirion SGP40 datasheet):
  "Measure raw signal" command, temperature/humidity-compensated:
    write: [0x26, 0x0F, RH_ticks_hi, RH_ticks_lo, RH_crc,
                        T_ticks_hi,  T_ticks_lo,  T_crc]
    (wait >= 30ms for the measurement to complete)
    read:  [SRAW_hi, SRAW_lo, SRAW_crc]   (3 bytes)

  RH/T compensation ticks (per datasheet section 3.6):
    rh_ticks = round(humidity_pct    * 65535 / 100)
    t_ticks  = round((temp_c + 45)   * 65535 / 175)

  CRC-8: polynomial 0x31, initial value 0xFF, computed over each 2-byte
  word independently. This is the same CRC Sensirion uses across their
  whole I2C sensor line (SHT3x/4x, SCD4x, etc).

WHY THIS DEPENDS ON sensirion-gas-index-algorithm INSTEAD OF HAND-ROLLING
THE VOC INDEX CONVERSION:
  What SGP40 returns over I2C is a raw, uncalibrated tick count (SRAW_VOC),
  not a usable VOC index. Converting that into the 0-500 "VOC Index" scale
  requires Sensirion's proprietary, stateful gas-index algorithm (it
  maintains a running baseline and expects roughly 1Hz continuous
  sampling to converge correctly) -- Sensirion does not publish the
  algorithm's internals for reimplementation, only distributes it as a
  library (`sensirion-gas-index-algorithm` on PyPI, official). Do not
  attempt to approximate this with a home-rolled formula; the raw ticks
  alone are not comparable across sensors or over time without it.

  If you'd rather avoid this dependency, this module also exposes
  read_raw() so you can log the raw SRAW_VOC tick count directly as a
  feature instead of an index -- just be aware raw ticks are sensor-
  instance-specific and not directly comparable/interpretable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from smbus2 import SMBus

SGP40_ADDR = 0x59
CRC8_POLY = 0x31
CRC8_INIT = 0xFF


def _crc8(data: bytes) -> int:
    crc = CRC8_INIT
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ CRC8_POLY) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def _word_with_crc(value: int) -> bytes:
    hi, lo = (value >> 8) & 0xFF, value & 0xFF
    return bytes([hi, lo, _crc8(bytes([hi, lo]))])


@dataclass
class SGP40Reading:
    voc_index: float
    sraw_voc: int
    timestamp: float


class SGP40Reader:
    def __init__(self, bus: int = 1, address: int = SGP40_ADDR):
        self._bus = SMBus(bus)
        self._addr = address
        try:
            from sensirion_gas_index_algorithm.voc_algorithm import VocAlgorithm
            self._voc_algo = VocAlgorithm()
        except ImportError:
            self._voc_algo = None  # read_raw() still works without this

    def close(self):
        self._bus.close()

    def read_raw(self, temp_c: float = 25.0, humidity_pct: float = 50.0) -> int | None:
        """Returns the raw SRAW_VOC tick count, or None on CRC failure.
        temp_c/humidity_pct should come from the DHT22 reading for a given
        cycle -- SGP40 has no onboard temp/hum sensor of its own."""
        rh_ticks = round(max(0.0, min(100.0, humidity_pct)) * 65535 / 100)
        t_ticks = round((temp_c + 45) * 65535 / 175)
        cmd = bytes([0x26, 0x0F]) + _word_with_crc(rh_ticks) + _word_with_crc(t_ticks)
        self._bus.write_i2c_block_data(self._addr, cmd[0], list(cmd[1:]))
        time.sleep(0.03)  # datasheet: >=30ms measurement duration
        resp = self._bus.read_i2c_block_data(self._addr, 0x00, 3)
        hi, lo, crc = resp
        if _crc8(bytes([hi, lo])) != crc:
            return None
        return (hi << 8) | lo

    def read(self, temp_c: float = 25.0, humidity_pct: float = 50.0) -> SGP40Reading | None:
        """Returns a calibrated VOC Index (0-500) via the Sensirion gas
        index algorithm. Requires sensirion-gas-index-algorithm to be
        installed AND to be called at a roughly steady ~1Hz cadence for
        the running baseline to converge -- do not call this once every
        60s expecting a meaningful index; see aggregator.py, which polls
        this at 1Hz internally and only reports the last value per window."""
        sraw = self.read_raw(temp_c=temp_c, humidity_pct=humidity_pct)
        if sraw is None:
            return None
        if self._voc_algo is None:
            raise RuntimeError(
                "sensirion-gas-index-algorithm not installed -- "
                "pip install sensirion-gas-index-algorithm, or use read_raw() instead."
            )
        index = self._voc_algo.process(sraw)
        return SGP40Reading(voc_index=float(index), sraw_voc=sraw, timestamp=time.time())


if __name__ == "__main__":
    r = SGP40Reader()
    print("Sampling at 1Hz -- VOC index needs a few dozen samples to converge...")
    for _ in range(60):
        print(r.read())
        time.sleep(1)
    r.close()
