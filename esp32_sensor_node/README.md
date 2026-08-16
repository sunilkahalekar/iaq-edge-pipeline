# ESP32 Sensor Node

Alternative to `../sensors/*.py` (direct-wired sensors on the Pi). Use
this if you want one ESP32 owning all four sensors and handing the Pi a
single serial JSON stream — solves the Pi 4's one-hardware-UART
limitation cleanly, since two of the four sensors (PMS5003, MH-Z19B) need
their own UART each.

**Pick one integration path per deployment, not both** — both write into
the same `sensor_readings` table via a script on the Pi; running both at
once would double-write or race.

## Wiring

```
                         ┌─────────────────────────┐
   PMS5003  TX ────────► │ GPIO4  (UART1 RX)        │
            (RX) ◄────── │ GPIO18 (UART1 TX, unused)│
                         │                          │
   MH-Z19B  TX ────────► │ GPIO16 (UART2 RX)        │
            RX  ◄──────  │ GPIO17 (UART2 TX)        │
                         │                          │        ESP32
   SGP40    SDA ◄───────►│ GPIO21 (I2C SDA)         │      WROOM-32
            SCL ◄───────►│ GPIO22 (I2C SCL)         │       DevKit
                         │                          │
   DHT22    DATA ◄──────►│ GPIO27                   │
            (+10k pull-up to 3.3V if module has none)│
                         │                          │
                         │ Onboard LED ── GPIO2      │  (heartbeat blink,
                         │                          │   once per 60s window)
                         │                     USB ──┼──► Raspberry Pi 4
                         └─────────────────────────┘      (/dev/ttyUSB0 or
                                                            /dev/ttyACM0)
```

All four sensors run off the ESP32's 3.3V/5V rails per their own
datasheets (PMS5003 and MH-Z19B typically want 5V, SGP40 and DHT22 want
3.3V — check your specific modules). None of the GPIO pins used here are
ESP32 strapping pins (0/2/5/12/15), chosen deliberately to avoid boot-mode
interference.

## Firmware setup

1. Install the **ESP32 board package** in Arduino IDE (Boards Manager →
   search "esp32", by Espressif Systems) if not already installed.
2. Install libraries via **Sketch → Include Library → Manage Libraries**:
   - `ArduinoJson` (Benoit Blanchon)
   - `DHT sensor library` (Adafruit) — accept the prompt to also install
     `Adafruit Unified Sensor`
   - `Sensirion I2C SGP40`
   - `Sensirion Gas Index Algorithm`
3. Open `esp32_sensor_node.ino`, select your board (Tools → Board → ESP32
   Dev Module) and port, and **before compiling**, open the Sensirion I2C
   SGP40 library's own example sketch (File → Examples → Sensirion I2C
   SGP40) and confirm `measureRawSignal()`'s exact parameter
   order/units match what's called in `pollSgp40()` — flagged explicitly
   at the top of the `.ino` file, since this couldn't be verified without
   network access when this was written.
4. Flash, then open the Serial Monitor at 115200 baud.

## Bench-testing before connecting to the Pi

Test one sensor at a time — don't wire all four and hope:

1. **PMS5003 alone**: power it, wait ~30s (fan warm-up), watch for valid
   JSON with real `pm1`/`pm2_5`/`pm10` values (not `null`) each window.
2. **MH-Z19B alone**: needs a 3-minute preheat after power-on. Breathe
   near it briefly — `co2` should rise, then decay back down.
3. **SGP40 alone**: `voc` starts near a baseline and should visibly react
   to isopropyl alcohol, hand sanitizer, or similar VOC source nearby.
   Needs a few dozen seconds of continuous sampling to converge — don't
   judge it from the first window.
4. **DHT22 alone**: `temp`/`hum` should read plausible values; occasional
   `null` is normal (see the `.ino` file's comment on this).
5. **All four together**: confirm every field populates across several
   consecutive windows, and that adding MH-Z19B's ~250ms blocking poll
   doesn't cause PMS5003 frames to go missing (check `pm1`/`pm2_5`/`pm10`
   stay non-null every window, not intermittently).

Only move to Pi integration once all four read plausible values
individually and together.

## Connecting to the Pi

```bash
ls /dev/ttyUSB* /dev/ttyACM*   # confirm the device path after plugging in
```
Then see `../iaq_sensors_esp32.py` and its systemd unit
`../iaq-sensors-esp32.service`.

## Status

**Not yet flashed or tested on real hardware.** The checksum/protocol
math for PMS5003 and MH-Z19B is a direct port of `../sensors/pms5003.py`
and `../sensors/mhz19b.py`, which WERE verified against known-good
datasheet test vectors in Python — the underlying math is the same, just
re-expressed in C++, so it should be correct, but this specific file has
not been compiled or run. The SGP40 call signature is explicitly flagged
above as needing verification against your installed library version.
Work through the bench-test checklist above before trusting this in
production.
