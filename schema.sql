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

-- ── Indexes for the queries you'll actually run (latest-first, date range) ──
CREATE INDEX IF NOT EXISTS idx_ct_ws  ON ct_vectors(window_start);
CREATE INDEX IF NOT EXISTS idx_sec_ts ON per_second_analytics(Timestamp);
CREATE INDEX IF NOT EXISTS idx_tel_ts ON tracking_telemetry(Timestamp_ISO8601);
CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw_detections(Timestamp_ISO8601);
CREATE INDEX IF NOT EXISTS idx_raw_class ON raw_detections(Class_Name);
