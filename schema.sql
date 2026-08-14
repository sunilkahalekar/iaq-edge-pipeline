-- schema.sql
-- ============================================================
-- IAQ pipeline database schema.
-- Apply once with:
--   sqlite3 /home/pi4/iaq_data/iaq.db < schema.sql
--
-- Re-running this is safe: every statement uses IF NOT EXISTS,
-- so applying it again on an existing database does nothing
-- destructive — it just confirms the schema is already there.
-- ============================================================

PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA wal_autocheckpoint = 2000;   -- SD card: fewer, larger checkpoints

-- ── Table 1: 60-second door/motion feature windows ─────────
CREATE TABLE IF NOT EXISTS ct_vectors (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    window_start   TEXT,
    M1_tau_open    INTEGER,
    M1_f_trans     INTEGER,
    M1_rho_open    REAL,
    M1_eps_max     INTEGER,
    M1_phi_open    REAL,
    M2_tau_open    INTEGER,
    M2_f_trans     INTEGER,
    M2_rho_open    REAL,
    M2_eps_max     INTEGER,
    M2_phi_open    REAL,
    M3_tau_open    INTEGER,
    M3_f_trans     INTEGER,
    M3_rho_open    REAL,
    M3_eps_max     INTEGER,
    M3_phi_open    REAL,
    n_person       INTEGER,
    mu_motion      REAL,
    sigma2_motion  REAL
);

-- ── Table 2: per-second summary (one row per second of video) ──
CREATE TABLE IF NOT EXISTS per_second_analytics (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    Timestamp                 TEXT,
    M1_State                  TEXT,
    M1_Toggles                INTEGER,
    M1_Setup_Sec              REAL,
    M1_Swap_Sec               REAL,
    M1_Raw_State              TEXT,
    M2_State                  TEXT,
    M2_Toggles                INTEGER,
    M2_Setup_Sec              REAL,
    M2_Swap_Sec               REAL,
    M2_Raw_State              TEXT,
    M3_State                  TEXT,
    M3_Toggles                INTEGER,
    M3_Setup_Sec              REAL,
    M3_Swap_Sec               REAL,
    M3_Raw_State              TEXT,
    Person_Count_Max          INTEGER,
    Unique_Total_Persons      INTEGER,
    Motion_Score_This_Second  REAL
);

-- ── Table 3: raw per-frame telemetry (highest volume table) ────
CREATE TABLE IF NOT EXISTS tracking_telemetry (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    Timestamp_ISO8601     TEXT,
    Unix_ms               INTEGER,
    M1_State_Debounced    TEXT,
    M2_State_Debounced    TEXT,
    M3_State_Debounced    TEXT,
    M1_State_Raw          TEXT,
    M2_State_Raw          TEXT,
    M3_State_Raw          TEXT,
    Zone_Count            INTEGER,
    Tracked_Person_IDs    TEXT,   -- JSON-encoded list, e.g. "[3, 7]"
    Global_Person_IDs     TEXT    -- JSON-encoded list, e.g. "[101, 104]"
);

-- ── Table 4: raw per-detection log (one row per detected object per frame) ──
CREATE TABLE IF NOT EXISTS raw_detections (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    Timestamp_ISO8601  TEXT,
    Unix_ms            INTEGER,
    Frame_Number       INTEGER,
    Class_Name         TEXT,
    Confidence         REAL,
    X1                 REAL,
    Y1                 REAL,
    X2                 REAL,
    Y2                 REAL
);

-- ── Table 5: 60-second pollutant sensor windows (written by iaq_sensors.py,
--    a separate process/service from the one that writes tables 1-4) ──
-- Column names match feature_engineering.py's build_base_columns() verbatim
-- (temp, hum, pm1, pm2_5, pm10, co2, voc). window_start here IS aligned to
-- wall-clock minute boundaries, but ct_vectors.window_start is NOT (it
-- drifts from Pi boot time) -- joining the two requires flooring both to
-- the minute at query time (see iaq_forecast.py), not an exact match.
CREATE TABLE IF NOT EXISTS sensor_readings (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    window_start   TEXT,
    pm1            REAL,
    pm2_5          REAL,
    pm10           REAL,
    co2            REAL,
    voc            REAL,
    temp           REAL,
    hum            REAL
);

-- ── Table 6: BiLSTM forecasts (written by iaq_forecast.py, a third
--    separate process/service) ──
-- predicted_for = predicted_at + lead_minutes. lead_minutes stored per row
-- (not assumed fixed) so it can change between model versions without
-- breaking historical rows' meaning.
CREATE TABLE IF NOT EXISTS forecasts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_at   TEXT,
    predicted_for  TEXT,
    lead_minutes   INTEGER,
    pm1            REAL,
    pm2_5          REAL,
    pm10           REAL,
    co2            REAL,
    voc            REAL
);

-- ── Indexes for the queries you'll actually run (latest-first, date range) ──
CREATE INDEX IF NOT EXISTS idx_ct_ws  ON ct_vectors(window_start);
CREATE INDEX IF NOT EXISTS idx_sec_ts ON per_second_analytics(Timestamp);
CREATE INDEX IF NOT EXISTS idx_tel_ts ON tracking_telemetry(Timestamp_ISO8601);
CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw_detections(Timestamp_ISO8601);
CREATE INDEX IF NOT EXISTS idx_raw_class ON raw_detections(Class_Name);
CREATE INDEX IF NOT EXISTS idx_sensor_ws ON sensor_readings(window_start);
CREATE INDEX IF NOT EXISTS idx_forecast_pf ON forecasts(predicted_for);
