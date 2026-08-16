/*
  esp32_sensor_node.ino
  ======================
  ESP32 firmware: reads PMS5003 (PM1/PM2.5/PM10), MH-Z19B (CO2), SGP40
  (VOC index), and DHT22 (temp/humidity), aggregates into one 60-second
  window, and emits ONE line of JSON over USB-Serial (Serial, UART0)
  every 60 seconds for the Raspberry Pi to consume.

  WHY AN ESP32 FRONT-END INSTEAD OF WIRING SENSORS DIRECTLY TO THE PI:
  The Raspberry Pi 4 has exactly one full-featured hardware UART exposed
  on its GPIO header (shared with Bluetooth by default) -- not enough
  for two UART sensors (PMS5003 + MH-Z19B) at once without disabling
  Bluetooth and accepting a single point of contention. The ESP32 has
  three independent hardware UARTs plus I2C plus GPIO, so it owns all
  four sensors with zero pin contention and hands the Pi one simple,
  self-contained JSON stream over a single USB cable.

  This is an ALTERNATIVE to the direct-wired Python drivers in
  ../sensors/*.py, not a replacement forced on you -- pick one
  integration path per deployment, not both, since both would try to
  write into the same sensor_readings table.

  PROTOCOL LOGIC HERE MIRRORS THE ALREADY-VERIFIED PYTHON VERSION in
  ../sensors/pms5003.py and ../sensors/mhz19b.py -- same checksum math
  (verified there against known-good datasheet values), same
  "atmospheric environment" PMS5003 field selection, same reasoning for
  using Sensirion's own gas-index algorithm library for SGP40 instead of
  a hand-rolled VOC formula. See those files' docstrings for the full
  protocol references.

  PIN MAP (ESP32-WROOM-32 DevKit -- adjust in the #defines below if your
  board/wiring differs). None of these are ESP32 strapping pins
  (0/2/5/12/15), chosen deliberately to avoid any boot-mode interference:
    UART1 RX  = GPIO4   <- PMS5003 TX
    UART1 TX  = GPIO18  -> PMS5003 RX (unused -- PMS5003 streams
                            unprompted in its default active mode; wired
                            anyway because Serial1.begin() requires a
                            TX pin argument even when it's never driven)
    UART2 RX  = GPIO16  <- MH-Z19B TX
    UART2 TX  = GPIO17  -> MH-Z19B RX
    I2C SDA   = GPIO21  <-> SGP40
    I2C SCL   = GPIO22  <-> SGP40
    DHT22     = GPIO27  <-> DHT22 data pin (+10k pull-up to 3.3V if your
                             module doesn't already have one on-board)
    STATUS_LED= GPIO2   -> onboard LED, blinks once per emitted window
                            as a simple "still alive" heartbeat

  REQUIRED LIBRARIES (Arduino IDE: Sketch -> Include Library -> Manage
  Libraries):
    - ArduinoJson (Benoit Blanchon)
    - DHT sensor library (Adafruit) + its Adafruit Unified Sensor dependency
    - Sensirion I2C SGP40
    - Sensirion Gas Index Algorithm

  *** VERIFY BEFORE COMPILING: pollSgp40()'s call to
  sgp40.measureRawSignal() below assumes the library takes plain
  %RH / degC compensation values and does tick conversion internally --
  this is the standard convention for Sensirion's official Arduino
  libraries, but library APIs do change across versions and this could
  not be checked against the actual installed library in the environment
  this file was written in (no ESP32/Arduino toolchain or network access
  available). Check the library's own example sketch (File -> Examples
  -> Sensirion I2C SGP40) and adjust the call signature to match before
  trusting the VOC output. ***

  NOT YET FLASHED OR TESTED ON REAL HARDWARE. The checksum/protocol math
  for PMS5003 and MH-Z19B mirrors Python code that WAS verified against
  datasheet test vectors (see ../sensors/*.py), but this C++ port has not
  itself been compiled or run on a device. Bench-test each sensor
  individually (see this folder's README.md) before trusting the
  aggregated output.
*/

#include <Arduino.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <SensirionI2cSgp40.h>
#include <VOCGasIndexAlgorithm.h>

// ── Pin configuration ────────────────────────────────────────────────────
#define PMS_RX_PIN   4
#define PMS_TX_PIN   18
#define MHZ_RX_PIN   16
#define MHZ_TX_PIN   17
#define I2C_SDA_PIN  21
#define I2C_SCL_PIN  22
#define DHT_PIN      27
#define STATUS_LED   2

#define PMS_BAUD     9600
#define MHZ_BAUD     9600

HardwareSerial PmsSerial(1);   // UART1
HardwareSerial MhzSerial(2);   // UART2
DHT dht(DHT_PIN, DHT22);
SensirionI2cSgp40 sgp40;
VOCGasIndexAlgorithm vocAlgorithm;

// ── Timing configuration ─────────────────────────────────────────────────
const unsigned long WINDOW_MS     = 60000;  // aggregate + emit every 60s
const unsigned long MHZ_POLL_MS   = 2000;   // MH-Z19B: no benefit polling faster
const unsigned long SGP40_POLL_MS = 1000;   // MUST stay ~1Hz -- gas index algorithm needs it
const unsigned long DHT_POLL_MS   = 2500;   // datasheet minimum is 2000ms

unsigned long lastWindowMs = 0;
unsigned long lastMhzMs = 0;
unsigned long lastSgp40Ms = 0;
unsigned long lastDhtMs = 0;

// ── Rolling buffers for the current 60s window ───────────────────────────
// Sized with margin above the expected sample counts per window
// (~60 PMS5003 frames/min, ~30 MH-Z19B samples/min, ~60 DHT22 samples/min
// at these poll intervals).
#define MAX_SAMPLES 120

float pm1Buf[MAX_SAMPLES], pm25Buf[MAX_SAMPLES], pm10Buf[MAX_SAMPLES];
int pmCount = 0;

float co2Buf[MAX_SAMPLES];
int co2Count = 0;

float tempBuf[MAX_SAMPLES], humBuf[MAX_SAMPLES];
int dhtCount = 0;

float lastVocIndex = NAN;     // last value, not averaged -- the gas-index
                               // algorithm's own internal state already
                               // smooths this; averaging already-smoothed
                               // values is redundant and can lag a real
                               // step change. Deliberately NOT reset
                               // between windows -- the algorithm's
                               // running baseline persists by design.

float lastKnownTemp = 25.0f;  // fallback SGP40 compensation values used
float lastKnownHum  = 50.0f;  // until the first real DHT22 reading arrives

// ── PMS5003 ───────────────────────────────────────────────────────────────
// 32-byte frame: 0x42 0x4D, length, then fields. Reads the "atmospheric
// environment" PM1/PM2.5/PM10 fields (bytes 10-15), NOT the "CF=1
// standard particle" fields (bytes 4-9) -- see ../sensors/pms5003.py's
// docstring for why that distinction matters.
void pollPms5003() {
  static uint8_t buf[32];
  static int idx = 0;

  while (PmsSerial.available()) {
    uint8_t b = PmsSerial.read();
    if (idx == 0 && b != 0x42) continue;
    if (idx == 1 && b != 0x4D) { idx = 0; continue; }
    buf[idx++] = b;
    if (idx == 32) {
      idx = 0;
      uint16_t checksumCalc = 0;
      for (int i = 0; i < 30; i++) checksumCalc += buf[i];
      uint16_t checksumRecv = (buf[30] << 8) | buf[31];
      if (checksumCalc != checksumRecv) continue;  // corrupt frame, drop

      float pm1  = (float)((buf[10] << 8) | buf[11]);
      float pm25 = (float)((buf[12] << 8) | buf[13]);
      float pm10 = (float)((buf[14] << 8) | buf[15]);

      if (pmCount < MAX_SAMPLES) {
        pm1Buf[pmCount] = pm1;
        pm25Buf[pmCount] = pm25;
        pm10Buf[pmCount] = pm10;
        pmCount++;
      }
    }
  }
}

// ── MH-Z19B ───────────────────────────────────────────────────────────────
// Command/response over UART2. Checksum: 0xFF - (sum(bytes[1..7]) & 0xFF) + 1
// -- identical formula to, and verified against the same known-good
// command-byte test case as, ../sensors/mhz19b.py.
uint8_t mhzChecksum(uint8_t *packet) {
  uint16_t sum = 0;
  for (int i = 1; i < 8; i++) sum += packet[i];
  return (uint8_t)(0xFF - (sum & 0xFF) + 1);
}

void pollMhz19b() {
  uint8_t cmd[9] = {0xFF, 0x01, 0x86, 0x00, 0x00, 0x00, 0x00, 0x00, 0x79};
  while (MhzSerial.available()) MhzSerial.read();  // clear stale bytes first
  MhzSerial.write(cmd, 9);

  // Short blocking wait for the 9-byte response -- MH-Z19B responds
  // within tens of ms normally; 250ms is a generous timeout without
  // risking a meaningful stall of pollPms5003()'s own UART servicing
  // (PMS5003's hardware UART has its own FIFO, sized below to tolerate
  // this brief pause -- see setup()'s setRxBufferSize call).
  unsigned long start = millis();
  int received = 0;
  uint8_t resp[9];
  while (millis() - start < 250 && received < 9) {
    if (MhzSerial.available()) resp[received++] = MhzSerial.read();
  }
  if (received != 9) return;                 // timeout, skip this cycle
  if (resp[0] != 0xFF || resp[1] != 0x86) return;
  if (mhzChecksum(resp) != resp[8]) return;   // corrupt response, drop

  float co2 = (float)((resp[2] << 8) | resp[3]);
  if (co2Count < MAX_SAMPLES) co2Buf[co2Count++] = co2;
}

// ── SGP40 (VOC index) ────────────────────────────────────────────────────
// Needs continuous ~1Hz sampling for the gas-index algorithm's running
// baseline to converge -- do not reduce SGP40_POLL_MS. Compensated with
// the latest DHT22 reading (or the fallback default until one arrives).
// *** See the file-level comment above about verifying this call's exact
// signature against your installed library version before trusting it. ***
void pollSgp40() {
  uint16_t srawVoc = 0;
  int16_t err = sgp40.measureRawSignal(lastKnownHum, lastKnownTemp, srawVoc);
  if (err != 0) return;

  int32_t vocIndex = vocAlgorithm.process((int32_t)srawVoc);
  lastVocIndex = (float)vocIndex;
}

// ── DHT22 ─────────────────────────────────────────────────────────────────
void pollDht22() {
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  if (isnan(h) || isnan(t)) return;   // normal occasional failure, not fatal
                                       // -- see ../sensors/dht22.py's
                                       // docstring, same sensor/protocol.

  lastKnownTemp = t;
  lastKnownHum = h;
  if (dhtCount < MAX_SAMPLES) {
    tempBuf[dhtCount] = t;
    humBuf[dhtCount] = h;
    dhtCount++;
  }
}

// ── Aggregation + JSON emit ──────────────────────────────────────────────
// Field names match ../sensors/aggregator.py's output verbatim (temp,
// hum, pm1, pm2_5, pm10, co2, voc) -- and, one level further up,
// feature_engineering.py's build_base_columns() -- so nothing downstream
// needs a renaming/mapping layer regardless of which sensor-integration
// path (ESP32 or direct-wired) produced the row.
float meanOrNan(float *buf, int count) {
  if (count == 0) return NAN;
  float sum = 0;
  for (int i = 0; i < count; i++) sum += buf[i];
  return sum / count;
}

void emitWindow() {
  StaticJsonDocument<256> doc;

  float pm1  = meanOrNan(pm1Buf, pmCount);
  float pm25 = meanOrNan(pm25Buf, pmCount);
  float pm10 = meanOrNan(pm10Buf, pmCount);
  float co2  = meanOrNan(co2Buf, co2Count);
  float temp = meanOrNan(tempBuf, dhtCount);
  float hum  = meanOrNan(humBuf, dhtCount);

  doc["pm1"]   = isnan(pm1)  ? (JsonVariant)nullptr : JsonVariant(pm1);
  doc["pm2_5"] = isnan(pm25) ? (JsonVariant)nullptr : JsonVariant(pm25);
  doc["pm10"]  = isnan(pm10) ? (JsonVariant)nullptr : JsonVariant(pm10);
  doc["co2"]   = isnan(co2)  ? (JsonVariant)nullptr : JsonVariant(co2);
  doc["voc"]   = isnan(lastVocIndex) ? (JsonVariant)nullptr : JsonVariant(lastVocIndex);
  doc["temp"]  = isnan(temp) ? (JsonVariant)nullptr : JsonVariant(temp);
  doc["hum"]   = isnan(hum)  ? (JsonVariant)nullptr : JsonVariant(hum);

  serializeJson(doc, Serial);
  Serial.println();

  // Reset window buffers -- lastVocIndex deliberately NOT reset here,
  // see its declaration comment above.
  pmCount = 0;
  co2Count = 0;
  dhtCount = 0;
}

// ── Setup / loop ──────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);          // USB link to the Pi -- this is what
                                  // shows up as /dev/ttyUSB0 or
                                  // /dev/ttyACM0 there
  PmsSerial.setRxBufferSize(512); // margin against the brief stall while
                                   // pollMhz19b() blocks waiting for a
                                   // response -- must be called before begin()
  PmsSerial.begin(PMS_BAUD, SERIAL_8N1, PMS_RX_PIN, PMS_TX_PIN);
  MhzSerial.begin(MHZ_BAUD, SERIAL_8N1, MHZ_RX_PIN, MHZ_TX_PIN);
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  dht.begin();
  sgp40.begin(Wire, SGP40_I2C_ADDR_59);
  vocAlgorithm.init();

  pinMode(STATUS_LED, OUTPUT);
  digitalWrite(STATUS_LED, LOW);

  unsigned long now = millis();
  lastWindowMs = now;
  lastMhzMs = now;
  lastSgp40Ms = now;
  lastDhtMs = now;
}

void loop() {
  unsigned long now = millis();

  pollPms5003();   // as fast as frames arrive, every loop iteration

  if (now - lastMhzMs >= MHZ_POLL_MS) {
    pollMhz19b();
    lastMhzMs = now;
  }
  if (now - lastSgp40Ms >= SGP40_POLL_MS) {
    pollSgp40();
    lastSgp40Ms = now;
  }
  if (now - lastDhtMs >= DHT_POLL_MS) {
    pollDht22();
    lastDhtMs = now;
  }
  if (now - lastWindowMs >= WINDOW_MS) {
    emitWindow();
    digitalWrite(STATUS_LED, !digitalRead(STATUS_LED));  // heartbeat blink
    lastWindowMs = now;
  }
}
