"""
db_writer.py
============
Local-database writer layer for the IAQ pipeline, replacing the original
AsyncCSVWriter / AsyncDictCSVWriter classes.

DEFAULT BACKEND: SQLite (single file, zero-config, correct for one Pi).

  - One writer thread + one queue per table (same pattern as the original
    CSV writers), so all sqlite3 access happens from a single thread.
    sqlite3 connections are NOT thread-safe by default — do not share
    a connection across threads. This module avoids that by construction.
  - WAL mode is enabled so a dashboard / read-only process can query the
    DB concurrently while the pipeline is still writing.
  - `synchronous=NORMAL` trades a small durability window (loses at most
    the last WAL frame on power loss) for much lower write latency —
    reasonable for a Pi writing once per second.

SWITCHING TO POSTGRES (if you want one central DB fed by multiple Pis):
  - Replace the `sqlite3.connect(...)` call in `_worker` with
    `psycopg2.connect(host=..., dbname=..., user=..., password=...)`
  - Replace the `?` placeholders in the INSERT statements with `%s`
  - Everything else (queue, thread, table schemas) stays the same.
"""

import sqlite3
import queue
import threading
import json
import time
from datetime import datetime

MACHINES = [1, 2, 3]

# ── Table schemas (mirrors the three original CSV files) ───────────────────

CT_COLUMNS = ["window_start"]
for _m in MACHINES:
    CT_COLUMNS += [
        f"M{_m}_tau_open", f"M{_m}_f_trans", f"M{_m}_rho_open",
        f"M{_m}_eps_max", f"M{_m}_phi_open",
    ]
CT_COLUMNS += ["n_person", "mu_motion", "sigma2_motion"]

SEC_COLUMNS = ["Timestamp"]
for _m in MACHINES:
    SEC_COLUMNS += [
        f"M{_m}_State", f"M{_m}_Toggles", f"M{_m}_Setup_Sec",
        f"M{_m}_Swap_Sec", f"M{_m}_Raw_State",
    ]
SEC_COLUMNS += ["Person_Count_Max", "Unique_Total_Persons", "Motion_Score_This_Second"]

TELEM_COLUMNS = [
    "Timestamp_ISO8601", "Unix_ms",
    "M1_State_Debounced", "M2_State_Debounced", "M3_State_Debounced",
    "M1_State_Raw", "M2_State_Raw", "M3_State_Raw",
    "Zone_Count", "Tracked_Person_IDs", "Global_Person_IDs",
]

# ── Raw per-detection log ───────────────────────────────────────────────────
# One row per detected object per inference call -- this is the simplest,
# most direct "what did the model see, and when" record, independent of
# the door-debouncing / windowing logic the other three tables encode.
RAW_DET_COLUMNS = [
    "Timestamp_ISO8601", "Unix_ms", "Frame_Number",
    "Class_Name", "Confidence", "X1", "Y1", "X2", "Y2",
]

# ── Table 5: 60-second pollutant sensor windows ─────────────────────────────
# Column names deliberately match what
# context-aware-bilstm/src/modeling/feature_engineering.py's
# build_base_columns() expects verbatim (temp, hum, pm1, pm2_5, pm10, co2,
# voc) -- see sensors/aggregator.py's docstring. window_start is aligned
# to wall-clock minute boundaries (naive local time, matching
# iaq_pipeline_pi.py's own clock convention) -- but ct_vectors.window_start
# is NOT minute-aligned, so joining these two tables requires flooring
# both to the minute at query time (iaq_forecast.py does this), not an
# exact window_start match.
SENSOR_COLUMNS = [
    "window_start", "pm1", "pm2_5", "pm10", "co2", "voc", "temp", "hum",
]

# ── Table 6: BiLSTM forecasts, written by iaq_forecast.py (own process) ────
# predicted_at: when this prediction was made (naive local time, matching
# every other timestamp column in this DB). predicted_for: predicted_at +
# lead_minutes -- the time this row's pm1/pm2_5/pm10/co2/voc values are a
# forecast FOR, not a reading of. Kept as separate columns (not just
# predicted_at + a fixed lead assumed elsewhere) so lead_minutes can change
# between model versions without breaking historical rows' meaning.
FORECAST_COLUMNS = [
    "predicted_at", "predicted_for", "lead_minutes",
    "pm1", "pm2_5", "pm10", "co2", "voc",
]

# Column -> SQLite type affinity. Anything not listed here defaults to TEXT.
_NUMERIC_INT = {
    "M1_tau_open", "M2_tau_open", "M3_tau_open",
    "M1_f_trans", "M2_f_trans", "M3_f_trans",
    "M1_eps_max", "M2_eps_max", "M3_eps_max",
    "n_person", "Unix_ms", "Zone_Count",
    "M1_Toggles", "M2_Toggles", "M3_Toggles",
    "Person_Count_Max", "Unique_Total_Persons",
    "Frame_Number", "lead_minutes",
}
_NUMERIC_REAL = {
    "M1_rho_open", "M2_rho_open", "M3_rho_open",
    "M1_phi_open", "M2_phi_open", "M3_phi_open",
    "mu_motion", "sigma2_motion",
    "M1_Setup_Sec", "M2_Setup_Sec", "M3_Setup_Sec",
    "M1_Swap_Sec", "M2_Swap_Sec", "M3_Swap_Sec",
    "Motion_Score_This_Second",
    "Confidence", "X1", "Y1", "X2", "Y2",
    "pm1", "pm2_5", "pm10", "co2", "voc", "temp", "hum",
}


def _sql_type(col: str) -> str:
    if col in _NUMERIC_INT:
        return "INTEGER"
    if col in _NUMERIC_REAL:
        return "REAL"
    return "TEXT"


def _create_table_sql(table: str, columns: list) -> str:
    cols_sql = ",\n    ".join(f'"{c}" {_sql_type(c)}' for c in columns)
    return (
        f'CREATE TABLE IF NOT EXISTS {table} (\n'
        f'    id INTEGER PRIMARY KEY AUTOINCREMENT,\n'
        f'    {cols_sql}\n'
        f');'
    )


CT_TABLE = "ct_vectors"
SEC_TABLE = "per_second_analytics"
TELEM_TABLE = "tracking_telemetry"
RAW_DET_TABLE = "raw_detections"
SENSOR_TABLE = "sensor_readings"
FORECAST_TABLE = "forecasts"


def init_db(db_path: str):
    """Create the DB file (if needed) and all six tables. Call once at
    startup -- safe to call from iaq_pipeline_pi.py, iaq_sensors.py, AND
    iaq_forecast.py (IF NOT EXISTS), since any of the three might start
    first."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    # PI (SD CARD): checkpoint every ~2000 WAL pages instead of the default
    # ~1000 — fewer, larger checkpoint fsyncs. WAL file will grow a bit
    # larger between checkpoints (a few MB at most for this row volume).
    conn.execute("PRAGMA wal_autocheckpoint=2000;")
    conn.execute(_create_table_sql(CT_TABLE, CT_COLUMNS))
    conn.execute(_create_table_sql(SEC_TABLE, SEC_COLUMNS))
    conn.execute(_create_table_sql(TELEM_TABLE, TELEM_COLUMNS))
    conn.execute(_create_table_sql(RAW_DET_TABLE, RAW_DET_COLUMNS))
    conn.execute(_create_table_sql(SENSOR_TABLE, SENSOR_COLUMNS))
    conn.execute(_create_table_sql(FORECAST_TABLE, FORECAST_COLUMNS))
    # Helpful indexes for common queries (latest-first, per-window lookups)
    conn.execute(f'CREATE INDEX IF NOT EXISTS idx_ct_ws ON {CT_TABLE}(window_start);')
    conn.execute(f'CREATE INDEX IF NOT EXISTS idx_sec_ts ON {SEC_TABLE}(Timestamp);')
    conn.execute(f'CREATE INDEX IF NOT EXISTS idx_tel_ts ON {TELEM_TABLE}(Timestamp_ISO8601);')
    conn.execute(f'CREATE INDEX IF NOT EXISTS idx_raw_ts ON {RAW_DET_TABLE}(Timestamp_ISO8601);')
    conn.execute(f'CREATE INDEX IF NOT EXISTS idx_raw_class ON {RAW_DET_TABLE}(Class_Name);')
    conn.execute(f'CREATE INDEX IF NOT EXISTS idx_sensor_ws ON {SENSOR_TABLE}(window_start);')
    conn.execute(f'CREATE INDEX IF NOT EXISTS idx_forecast_pf ON {FORECAST_TABLE}(predicted_for);')
    conn.commit()
    conn.close()


class AsyncDBWriter:
    """
    ONE background thread, ONE sqlite3 connection, serving ALL tables.

    WHY THIS REPLACED PER-TABLE CONNECTIONS:
    SQLite's WAL mode allows only one writer to hold the file's write lock
    at a time -- across the WHOLE file, not per table. The earlier design
    gave each table its own connection/thread, each independently opening
    a transaction on INSERT and not committing until either 200 rows piled
    up or its queue went fully idle for commit_every_sec. At low, steady
    row rates (rows trickling in slower than 200/batch but faster than the
    idle gap), a writer's transaction could stay open indefinitely --
    holding the DB-wide write lock and starving every other writer past
    its busy_timeout, producing "database is locked".

    This version has exactly one connection, so there is only ever one
    thing that could hold the write lock. Commits are now driven by a
    real wall-clock deadline (checked every ~1s) instead of "queue went
    idle", so a transaction can never stay open longer than
    commit_every_sec regardless of how steadily rows arrive.
    """

    def __init__(self, db_path: str, commit_every_sec: float = 10.0,
                 batch_size: int = 200, maxsize: int = 16384):
        self._db_path = db_path
        self._commit_every_sec = commit_every_sec
        self._batch_size = batch_size
        self._q = queue.Queue(maxsize=maxsize)
        self._insert_sql_cache = {}
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _insert_sql(self, table: str, columns: list) -> str:
        sql = self._insert_sql_cache.get(table)
        if sql is None:
            placeholders = ", ".join(["?"] * len(columns))
            col_list = ", ".join(f'"{c}"' for c in columns)
            sql = f'INSERT INTO {table} ({col_list}) VALUES ({placeholders})'
            self._insert_sql_cache[table] = sql
        return sql

    @staticmethod
    def _row_to_values(row: dict, columns: list):
        vals = []
        for c in columns:
            v = row.get(c)
            if isinstance(v, (list, dict)):
                v = json.dumps(v)
            elif isinstance(v, datetime):
                v = v.strftime("%Y-%m-%d %H:%M:%S")
            vals.append(v)
        return tuple(vals)

    def _worker(self):
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA wal_autocheckpoint=2000;")
        cur = conn.cursor()
        pending = 0
        last_commit = time.monotonic()

        while True:
            try:
                item = self._q.get(timeout=1.0)   # short poll so the
                                                    # deadline check below
                                                    # always runs at least
                                                    # once per second
            except queue.Empty:
                item = "__TICK__"

            if item is None:   # sentinel: shut down
                if pending:
                    conn.commit()
                break

            if item != "__TICK__":
                table, columns, row = item
                sql = self._insert_sql(table, columns)
                cur.execute(sql, self._row_to_values(row, columns))
                pending += 1

            now = time.monotonic()
            deadline_hit = (now - last_commit) >= self._commit_every_sec
            if pending and (pending >= self._batch_size or deadline_hit):
                conn.commit()
                pending = 0
                last_commit = now

        conn.close()

    def write(self, table: str, columns: list, row: dict):
        self._q.put((table, columns, row))

    def close(self):
        self._q.put(None)
        self._t.join()


class TableWriterHandle:
    """
    Thin per-table view onto a shared AsyncDBWriter, so calling code can
    keep doing `ct_writer.write(row)` without knowing about the other
    tables sharing the same underlying connection. `.close()` is a no-op
    here on purpose -- only the shared AsyncDBWriter actually owns the
    thread/connection; close that once, via `close_writers()`.
    """

    def __init__(self, shared_writer: AsyncDBWriter, table: str, columns: list):
        self._shared = shared_writer
        self._table = table
        self._columns = columns

    def write(self, row: dict):
        self._shared.write(self._table, self._columns, row)

    def close(self):
        pass   # intentionally does nothing -- see close_writers()


def make_writers(db_path: str, commit_every_sec: float = 10.0):
    """
    Initializes the schema and returns:
        (ct_writer, sec_writer, tel_writer, det_writer, shared_writer)

    Use the first four exactly as before (ct_writer.write(row), etc).
    Call shared_writer.close() exactly ONCE at shutdown -- NOT the
    individual handles' .close(), which are no-ops by design.
    """
    init_db(db_path)
    shared = AsyncDBWriter(db_path, commit_every_sec=commit_every_sec)
    ct_writer = TableWriterHandle(shared, CT_TABLE, CT_COLUMNS)
    sec_writer = TableWriterHandle(shared, SEC_TABLE, SEC_COLUMNS)
    tel_writer = TableWriterHandle(shared, TELEM_TABLE, TELEM_COLUMNS)
    det_writer = TableWriterHandle(shared, RAW_DET_TABLE, RAW_DET_COLUMNS)
    return ct_writer, sec_writer, tel_writer, det_writer, shared


def make_sensor_writer(db_path: str, commit_every_sec: float = 10.0):
    """
    Separate from make_writers() on purpose: iaq_sensors.py runs as its own
    process (own systemd unit), not a thread inside iaq_pipeline_pi.py, so
    it needs its own AsyncDBWriter instance -- extending make_writers()'s
    return tuple would have broken iaq_pipeline_pi.py's existing unpacking
    call. Two independent single-writer-thread processes sharing one
    SQLite file (both WAL mode, both with 30s busy timeout, both batching
    commits) is safe -- this is NOT the "one connection per table within
    one process" pattern that caused the original "database is locked"
    bug (see AsyncDBWriter's docstring); it's two well-behaved writers
    that will occasionally, briefly serialize on the file lock, which the
    busy timeout already covers. Sensor writes happen once per 60s, so
    that contention window is rare and short.

    Returns (sensor_writer, shared_writer). Call shared_writer.close()
    once at shutdown.
    """
    init_db(db_path)
    shared = AsyncDBWriter(db_path, commit_every_sec=commit_every_sec)
    sensor_writer = TableWriterHandle(shared, SENSOR_TABLE, SENSOR_COLUMNS)
    return sensor_writer, shared


def make_forecast_writer(db_path: str, commit_every_sec: float = 5.0):
    """
    Third independent process/writer, same reasoning as
    make_sensor_writer() above. Forecasts happen once per minute (lower
    volume than either camera or sensor writes), so contention risk is
    even smaller than the sensor writer's already-small one. Shorter
    default commit interval than make_sensor_writer's (5s vs 10s) since a
    forecast is only useful if it shows up promptly -- there's no SD-card-
    wear argument for batching harder here, write volume is tiny.

    Returns (forecast_writer, shared_writer). Call shared_writer.close()
    once at shutdown.
    """
    init_db(db_path)
    shared = AsyncDBWriter(db_path, commit_every_sec=commit_every_sec)
    forecast_writer = TableWriterHandle(shared, FORECAST_TABLE, FORECAST_COLUMNS)
    return forecast_writer, shared
