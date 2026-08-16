"""
dashboard.py — live HTML dashboard for the IAQ database.

Shows the two things that matter operationally at a glance:
  1. Door state per machine (Open / Closed), with how long it's been in
     that state and today's toggle count.
  2. Person count — current people in the zone, and total unique people
     seen this session.

Reads the SQLite DB in READ-ONLY mode, so it can run continuously
alongside the pipeline's writer without any risk of write contention.

USAGE
    python3 dashboard.py --db /home/pi4/iaq_data/iaq.db --port 8001

Then from any device on the network:
    http://<PI_IP_ADDRESS>:8001
"""

import argparse
import sqlite3
import time
from flask import Flask, jsonify, render_template_string

DB_PATH = "/home/pi4/iaq_data/iaq.db"
STREAM_PORT = 8000   # must match STREAM_PORT in iaq_pipeline_pi.py
MACHINES = [1, 2, 3]

# ── Air-quality classification bands ────────────────────────────────────
# These are commonly-cited REFERENCE bands, not a substitute for your
# facility's actual applicable regulation. PM1/PM2.5/PM10 bands follow the
# US EPA's published AQI breakpoints (µg/m³, widely used as a general
# reference even outside the US). CO2 bands follow commonly-cited indoor
# air quality guidance (ASHRAE-adjacent educational bands, NOT the same
# as an 8-hour occupational exposure limit like OSHA's 5000ppm PEL --
# those are for a different purpose, sustained workplace exposure, not
# instantaneous indoor-air comfort/alertness). VOC uses Sensirion's own
# published index scale for the SGP40 (0-500, baseline/typical ~100).
# CONFIRM the applicable standard for your jurisdiction/industry with
# your safety officer before treating this color-coding as a compliance
# signal -- it's a dashboard view, not a certified safety instrument.
THRESHOLDS = {
    # pollutant: [(upper_bound, label, css_class), ...] ascending, last entry is the ceiling
    "pm1":   [(12, "Good", "ok"), (35, "Moderate", "warn"), (55, "Unhealthy (sensitive)", "warn"), (150, "Unhealthy", "danger"), (9e9, "Very Unhealthy", "danger")],
    "pm2_5": [(12, "Good", "ok"), (35, "Moderate", "warn"), (55, "Unhealthy (sensitive)", "warn"), (150, "Unhealthy", "danger"), (9e9, "Very Unhealthy", "danger")],
    "pm10":  [(54, "Good", "ok"), (154, "Moderate", "warn"), (254, "Unhealthy (sensitive)", "warn"), (354, "Unhealthy", "danger"), (9e9, "Very Unhealthy", "danger")],
    "co2":   [(800, "Excellent", "ok"), (1000, "Good", "ok"), (1500, "Moderate", "warn"), (2500, "Poor", "danger"), (9e9, "Very Poor", "danger")],
    "voc":   [(100, "Baseline/Low", "ok"), (200, "Elevated", "warn"), (400, "High", "danger"), (9e9, "Very High", "danger")],
}
POLLUTANT_UNITS = {"pm1": "µg/m³", "pm2_5": "µg/m³", "pm10": "µg/m³", "co2": "ppm", "voc": "index", "temp": "°C", "hum": "%"}
POLLUTANT_LABELS = {"pm1": "PM1.0", "pm2_5": "PM2.5", "pm10": "PM10", "co2": "CO₂", "voc": "VOC Index", "temp": "Temperature", "hum": "Humidity"}


def classify_value(pollutant: str, value):
    """Returns (label, css_class) for a pollutant reading, or (None, None)
    if the pollutant has no configured bands (temp/hum) or value is None."""
    if value is None or pollutant not in THRESHOLDS:
        return None, None
    for upper, label, css_class in THRESHOLDS[pollutant]:
        if value <= upper:
            return label, css_class
    return "Unknown", "warn"


def worst_class(classes):
    """danger > warn > ok, for rolling several pollutants' status into one
    overall badge."""
    if "danger" in classes:
        return "danger"
    if "warn" in classes:
        return "warn"
    if "ok" in classes:
        return "ok"
    return None


def get_conn():
    # Read-only URI connection: never blocks or contends with the pipeline's
    # writer thread, and can't accidentally write to the DB from this process.
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)


def compute_pipeline_health(latest_tel, conn):
    """
    Answers "is the YOLO/camera pipeline actually running right now, and
    how fast is it really processing frames" -- not just "is there data
    in the table at all". This is what makes inference status visible on
    the dashboard, not just historical presence of rows.
    """
    if latest_tel is None:
        return {"state": "no_data", "seconds_since_last": None, "avg_frame_interval_sec": None}

    now_ms = time.time() * 1000
    age_sec = (now_ms - latest_tel["Unix_ms"]) / 1000.0 if latest_tel["Unix_ms"] else None

    # Average real interval between the last 10 telemetry rows -- this is
    # your actual achieved frame rate on this Pi, not the configured target.
    recent = conn.execute(
        "SELECT Unix_ms FROM tracking_telemetry ORDER BY id DESC LIMIT 10"
    ).fetchall()
    intervals = []
    for i in range(len(recent) - 1):
        a, b = recent[i]["Unix_ms"], recent[i + 1]["Unix_ms"]
        if a and b:
            intervals.append((a - b) / 1000.0)
    avg_interval = round(sum(intervals) / len(intervals), 2) if intervals else None

    if age_sec is None:
        state = "no_data"
    elif age_sec < 15:
        state = "running"
    else:
        state = "stale"

    return {
        "state": state,
        "seconds_since_last": round(age_sec, 1) if age_sec is not None else None,
        "avg_frame_interval_sec": avg_interval,
    }


def fetch_status():
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    try:
        latest_tel = conn.execute(
            "SELECT * FROM tracking_telemetry ORDER BY id DESC LIMIT 1"
        ).fetchone()

        latest_sec = conn.execute(
            "SELECT * FROM per_second_analytics ORDER BY id DESC LIMIT 1"
        ).fetchone()

        recent_rows = conn.execute(
            "SELECT Timestamp_ISO8601, M1_State_Debounced, M2_State_Debounced, "
            "M3_State_Debounced, Zone_Count FROM tracking_telemetry "
            "ORDER BY id DESC LIMIT 15"
        ).fetchall()

        pipeline_health = compute_pipeline_health(latest_tel, conn)
    finally:
        conn.close()

    machines = {}
    for m in MACHINES:
        state = latest_tel[f"M{m}_State_Debounced"] if latest_tel else "Unknown"
        toggles = latest_sec[f"M{m}_Toggles"] if latest_sec else 0
        setup_sec = latest_sec[f"M{m}_Setup_Sec"] if latest_sec else 0
        swap_sec = latest_sec[f"M{m}_Swap_Sec"] if latest_sec else 0
        machines[m] = {
            "state": state,
            "toggles": toggles,
            "setup_sec": round(setup_sec, 1) if setup_sec is not None else 0,
            "swap_sec": round(swap_sec, 1) if swap_sec is not None else 0,
        }

    return {
        "last_updated": latest_tel["Timestamp_ISO8601"] if latest_tel else None,
        "zone_count": latest_tel["Zone_Count"] if latest_tel else 0,
        "unique_total": latest_sec["Unique_Total_Persons"] if latest_sec else 0,
        "machines": machines,
        "recent_rows": [dict(r) for r in recent_rows],
        "pipeline_health": pipeline_health,
    }


def fetch_history(start: str, end: str, limit: int = 5000):
    """
    Returns per_second_analytics rows between two 'YYYY-MM-DD HH:MM:SS'
    strings (inclusive). Column is stored in exactly that sortable format,
    so a plain string BETWEEN works correctly without any date parsing.
    """
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT Timestamp, Person_Count_Max, M1_State, M2_State, M3_State "
            "FROM per_second_analytics "
            "WHERE Timestamp BETWEEN ? AND ? "
            "ORDER BY Timestamp ASC LIMIT ?",
            (start, end, limit),
        ).fetchall()
    finally:
        conn.close()

    return {
        "labels": [r["Timestamp"] for r in rows],
        "person_count": [r["Person_Count_Max"] for r in rows],
        "m1_open": [1 if r["M1_State"] == "Open" else 0 for r in rows],
        "m2_open": [1 if r["M2_State"] == "Open" else 0 for r in rows],
        "m3_open": [1 if r["M3_State"] == "Open" else 0 for r in rows],
        "row_count": len(rows),
        "resolution": "second",
    }


def fetch_history_minute(start: str, end: str, limit: int = 2000):
    """
    Per-minute resolution, reading the ALREADY-AGGREGATED ct_vectors table
    (one row per 60-second window) instead of raw per-second rows. Far
    better signal-to-noise for anything longer than a few minutes --
    n_person is the max concurrent count in that minute, M{m}_rho_open is
    the FRACTION of that minute the door was open (0.0-1.0), not a raw
    0/1 flag, which is a more honest summary at minute granularity.
    """
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT window_start, n_person, M1_rho_open, M2_rho_open, M3_rho_open "
            "FROM ct_vectors WHERE window_start BETWEEN ? AND ? "
            "ORDER BY window_start ASC LIMIT ?",
            (start, end, limit),
        ).fetchall()
    finally:
        conn.close()

    return {
        "labels": [r["window_start"] for r in rows],
        "person_count": [r["n_person"] for r in rows],
        "m1_open": [r["M1_rho_open"] for r in rows],
        "m2_open": [r["M2_rho_open"] for r in rows],
        "m3_open": [r["M3_rho_open"] for r in rows],
        "row_count": len(rows),
        "resolution": "minute",
    }


POLLUTANTS = ["pm1", "pm2_5", "pm10", "co2", "voc"]


def _service_health(latest_ts_str, stale_after_sec=150):
    """Same reasoning as compute_pipeline_health() above, generalized for
    any table with a sortable 'YYYY-MM-DD HH:MM:SS' timestamp column and a
    ~60s expected write cadence -- sensor_readings and forecasts both
    qualify. stale_after_sec defaults to 2.5x the ~60s cadence, giving
    margin for a slightly late write without false-alarming."""
    if latest_ts_str is None:
        return {"state": "no_data", "seconds_since_last": None}
    try:
        from datetime import datetime as _dt
        age_sec = (_dt.now() - _dt.strptime(latest_ts_str, "%Y-%m-%d %H:%M:%S")).total_seconds()
    except ValueError:
        return {"state": "no_data", "seconds_since_last": None}
    state = "running" if age_sec < stale_after_sec else "stale"
    return {"state": state, "seconds_since_last": round(age_sec, 1)}


def fetch_air_quality():
    """Latest sensor_readings row (current conditions) + latest forecasts
    row (10-minutes-ahead prediction) + health of both ingestion paths.
    Returns per-pollutant classification so the frontend doesn't need to
    duplicate THRESHOLDS logic in JS."""
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    try:
        latest_sensor = conn.execute(
            "SELECT * FROM sensor_readings ORDER BY id DESC LIMIT 1"
        ).fetchone()
        latest_forecast = conn.execute(
            "SELECT * FROM forecasts ORDER BY id DESC LIMIT 1"
        ).fetchone()
    except sqlite3.OperationalError:
        # sensor_readings/forecasts tables don't exist yet -- a Pi running
        # only the camera pipeline (steps 1-12) without the optional
        # sensor/forecast services (13-14) installed. Not an error.
        conn.close()
        return {
            "available": False,
            "sensor_health": {"state": "not_installed", "seconds_since_last": None},
            "forecast_health": {"state": "not_installed", "seconds_since_last": None},
        }
    finally:
        conn.close()

    sensor_health = _service_health(latest_sensor["window_start"] if latest_sensor else None)
    forecast_health = _service_health(latest_forecast["predicted_at"] if latest_forecast else None)

    current = {}
    classified = {}
    for p in POLLUTANTS:
        val = latest_sensor[p] if latest_sensor else None
        current[p] = val
        label, css_class = classify_value(p, val)
        classified[p] = {"label": label, "class": css_class}
    current["temp"] = latest_sensor["temp"] if latest_sensor else None
    current["hum"] = latest_sensor["hum"] if latest_sensor else None

    predicted = None
    predicted_classified = {}
    if latest_forecast:
        predicted = {p: latest_forecast[p] for p in POLLUTANTS}
        predicted["predicted_for"] = latest_forecast["predicted_for"]
        predicted["predicted_at"] = latest_forecast["predicted_at"]
        predicted["lead_minutes"] = latest_forecast["lead_minutes"]
        for p in POLLUTANTS:
            label, css_class = classify_value(p, latest_forecast[p])
            predicted_classified[p] = {"label": label, "class": css_class}

    overall_current = worst_class([c["class"] for c in classified.values() if c["class"]])
    overall_predicted = worst_class([c["class"] for c in predicted_classified.values() if c["class"]]) if predicted else None

    return {
        "available": True,
        "current": current,
        "current_classified": classified,
        "current_at": latest_sensor["window_start"] if latest_sensor else None,
        "predicted": predicted,
        "predicted_classified": predicted_classified,
        "overall_current": overall_current,
        "overall_predicted": overall_predicted,
        "sensor_health": sensor_health,
        "forecast_health": forecast_health,
    }


def fetch_air_quality_history(start: str, end: str, limit: int = 2000):
    """Actual sensor_readings + forecasted values, both keyed by the
    timestamp the reading/prediction is FOR (window_start / predicted_for
    respectively) -- so overlaying them on one chart, aligned by that
    shared x-axis, directly shows how far ahead of the actual reading the
    prediction landed. Returns a UNIONED, sorted label set with nulls
    where a series has no point at that timestamp (see dashboard's JS:
    Chart.js renders a gap for null points in a line dataset, no time-
    scale adapter library needed)."""
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    try:
        actual_rows = conn.execute(
            "SELECT window_start AS ts, pm1, pm2_5, pm10, co2, voc FROM sensor_readings "
            "WHERE window_start BETWEEN ? AND ? ORDER BY window_start ASC LIMIT ?",
            (start, end, limit),
        ).fetchall()
        pred_rows = conn.execute(
            "SELECT predicted_for AS ts, pm1, pm2_5, pm10, co2, voc FROM forecasts "
            "WHERE predicted_for BETWEEN ? AND ? ORDER BY predicted_for ASC LIMIT ?",
            (start, end, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return {"available": False}
    finally:
        conn.close()

    actual_by_ts = {r["ts"]: r for r in actual_rows}
    pred_by_ts = {r["ts"]: r for r in pred_rows}
    all_ts = sorted(set(actual_by_ts) | set(pred_by_ts))

    result = {"available": True, "labels": all_ts, "row_count": len(all_ts)}
    for p in POLLUTANTS:
        result[f"actual_{p}"] = [actual_by_ts[t][p] if t in actual_by_ts else None for t in all_ts]
        result[f"predicted_{p}"] = [pred_by_ts[t][p] if t in pred_by_ts else None for t in all_ts]
    return result


INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>IAQ Live Dashboard</title>
<script src="/static/chart.umd.js"></script>
<style>
  :root {
    --bg: #0f1115; --card: #1a1d24; --border: #2a2e38;
    --text: #e8eaed; --muted: #8a8f9b;
    --open: #ffbe46; --closed: #4ea1ff; --accent: #4ade80;
  }
  * { box-sizing: border-box; }
  body {
    background: radial-gradient(ellipse at top, #14161c 0%, var(--bg) 60%);
    color: var(--text);
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 28px; min-height: 100vh;
  }
  h1 { font-size: 21px; font-weight: 700; margin: 0 0 4px; letter-spacing: -0.01em; }
  .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 24px; display: flex; align-items: center; gap: 6px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 20px; }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2), 0 8px 24px -12px rgba(0,0,0,0.4);
    transition: border-color 0.2s ease, transform 0.15s ease;
  }
  .card:hover { border-color: #363c4a; }
  .card h3 { margin: 0 0 12px; font-size: 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
  .door-state {
    display: inline-block; padding: 6px 14px; border-radius: 8px;
    font-weight: 700; font-size: 18px; margin-bottom: 8px;
  }
  .door-state.open { background: rgba(255,190,70,0.15); color: var(--open); }
  .door-state.closed { background: rgba(78,161,255,0.15); color: var(--closed); }
  .door-meta { color: var(--muted); font-size: 12px; line-height: 1.6; }
  .big-number { font-size: 42px; font-weight: 700; color: var(--accent); line-height: 1; letter-spacing: -0.02em; }
  .big-number-label { color: var(--muted); font-size: 12px; margin-top: 6px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { text-align: left; padding: 10px 10px; border-bottom: 1px solid var(--border); }
  tbody tr { transition: background 0.15s ease; }
  tbody tr:hover { background: rgba(255,255,255,0.02); }
  th { color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 10px; letter-spacing: 0.06em; }
  .pill { padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; }
  .pill.open { background: rgba(255,190,70,0.15); color: var(--open); }
  .pill.closed { background: rgba(78,161,255,0.15); color: var(--closed); }
  .stale-warning { color: #ff6b6b; font-size: 12px; margin-top: 4px; display: none; }
  .badge {
    display: inline-block; padding: 4px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 600;
  }
  .badge-running { background: rgba(74,222,128,0.15); color: var(--accent); }
  .badge-stale { background: rgba(255,190,70,0.15); color: var(--open); }
  .badge-nodata { background: rgba(255,107,107,0.15); color: #ff6b6b; }
  .js-error-banner {
    display: none; background: rgba(255,107,107,0.12); border: 1px solid #ff6b6b;
    color: #ff9a9a; padding: 10px 14px; border-radius: 8px; font-size: 12px;
    margin-bottom: 16px;
  }
  .section { margin-bottom: 22px; }
  .video-row { display: grid; grid-template-columns: 1.6fr 1fr; gap: 16px; margin-bottom: 22px; align-items: start; }
  @media (max-width: 900px) { .video-row { grid-template-columns: 1fr; } }

  .status-panel { display: flex; flex-direction: column; gap: 16px; }

  .people-card {
    background: linear-gradient(155deg, #1c2029 0%, #15171e 100%);
    border: 1px solid #2c3140;
    box-shadow: 0 1px 2px rgba(0,0,0,0.25), 0 12px 28px -14px rgba(74,222,128,0.12);
  }
  .people-card-header { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
  .people-card-header h3 { margin: 0; }
  .live-dot {
    width: 8px; height: 8px; border-radius: 50%; background: var(--accent);
    box-shadow: 0 0 0 0 rgba(74,222,128,0.6);
    animation: pulse-dot 2s infinite;
  }
  @keyframes pulse-dot {
    0%   { box-shadow: 0 0 0 0 rgba(74,222,128,0.55); }
    70%  { box-shadow: 0 0 0 8px rgba(74,222,128,0); }
    100% { box-shadow: 0 0 0 0 rgba(74,222,128,0); }
  }
  .people-card .big-number { font-size: 54px; background: linear-gradient(135deg, #4ade80, #86efac);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }

  #machine-cards .card { position: relative; padding-left: 24px; overflow: hidden; transition: padding-left 0.2s ease; }
  #machine-cards .card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: var(--closed); transition: background 0.3s ease;
  }
  #machine-cards .card:has(.door-state.open)::before { background: var(--open); }
  #machine-cards .card:hover { padding-left: 26px; }
  .video-card { padding: 14px; }
  .video-feed { width: 100%; border-radius: 10px; background: #000; display: block; min-height: 200px; }
  .video-error { display: none; color: #ff9a9a; font-size: 12px; margin-top: 8px; }

  .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 14px; }
  .preset-btn {
    background: var(--card); border: 1px solid var(--border); color: var(--text);
    padding: 7px 14px; border-radius: 8px; font-size: 12px; cursor: pointer;
  }
  .preset-btn:hover { border-color: var(--accent); }
  .preset-btn.active { background: rgba(74,222,128,0.15); border-color: var(--accent); color: var(--accent); }
  .range-form { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .range-form input {
    background: var(--card); border: 1px solid var(--border); color: var(--text);
    padding: 6px 10px; border-radius: 8px; font-size: 12px;
  }
  .apply-btn {
    background: var(--accent); border: none; color: #0f1115; font-weight: 700;
    padding: 7px 16px; border-radius: 8px; font-size: 12px; cursor: pointer;
  }
  .chart-wrap { position: relative; height: 320px; }
  .row-count { color: var(--muted); font-size: 11px; }

  .timeline-legend { display: flex; align-items: center; gap: 16px; margin-bottom: 12px; font-size: 12px; color: var(--muted); }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .legend-swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
  .swatch-closed { background: var(--closed); }
  .swatch-open { background: var(--open); }
  .legend-note { margin-left: auto; font-style: italic; }

  .timeline-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
  .timeline-label { width: 70px; font-size: 13px; font-weight: 600; color: var(--text); flex-shrink: 0; }
  .timeline-track-wrap { flex: 1; position: relative; }
  .timeline-track {
    width: 100%; height: 30px; border-radius: 8px; display: block;
    box-shadow: inset 0 0 0 1px var(--border);
  }
  .timeline-current {
    width: 64px; text-align: center; font-size: 11px; font-weight: 700;
    padding: 4px 0; border-radius: 6px; flex-shrink: 0;
  }
  .timeline-current.open { background: rgba(255,190,70,0.15); color: var(--open); }
  .timeline-current.closed { background: rgba(78,161,255,0.15); color: var(--closed); }
  .timeline-axis { display: flex; justify-content: space-between; color: var(--muted); font-size: 10px;
                    margin-top: 4px; padding-left: 82px; padding-right: 76px; }
  .timeline-tooltip {
    position: absolute; display: none; background: #000; color: #fff; font-size: 11px;
    padding: 4px 8px; border-radius: 6px; pointer-events: none; white-space: nowrap;
    transform: translate(-50%, -130%); z-index: 10;
  }

  /* ── Air quality / forecast / alert additions ────────────────────── */
  :root {
    --ok: #4ade80; --warn: #ffbe46; --danger: #ff5c5c;
  }
  .alert-banner {
    display: none; background: rgba(255,92,92,0.12); border: 1px solid var(--danger);
    color: #ff9a9a; padding: 14px 18px; border-radius: 12px; font-size: 13px;
    margin-bottom: 20px; align-items: center; gap: 10px;
  }
  .alert-banner.show { display: flex; }
  .alert-banner .alert-icon { font-size: 20px; }
  .alert-banner strong { color: #ffb3b3; }

  .badge-ok { background: rgba(74,222,128,0.15); color: var(--ok); }
  .badge-warn { background: rgba(255,190,70,0.15); color: var(--warn); }
  .badge-danger { background: rgba(255,92,92,0.15); color: var(--danger); }

  .aq-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; }
  .aq-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 14px 16px; position: relative; overflow: hidden;
  }
  .aq-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: var(--border);
  }
  .aq-card.status-ok::before { background: var(--ok); }
  .aq-card.status-warn::before { background: var(--warn); }
  .aq-card.status-danger::before { background: var(--danger); }
  .aq-card .aq-label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
  .aq-card .aq-value { font-size: 26px; font-weight: 700; line-height: 1; }
  .aq-card .aq-unit { color: var(--muted); font-size: 11px; margin-left: 4px; font-weight: 400; }
  .aq-card .aq-status { font-size: 11px; margin-top: 8px; font-weight: 600; }
  .aq-card.status-ok .aq-status { color: var(--ok); }
  .aq-card.status-warn .aq-status { color: var(--warn); }
  .aq-card.status-danger .aq-status { color: var(--danger); }
  .aq-card .aq-value.no-data { color: var(--muted); font-size: 16px; font-weight: 500; }

  .forecast-panel { display: grid; grid-template-columns: 1fr auto 1fr; gap: 16px; align-items: center; }
  .forecast-side { text-align: center; }
  .forecast-side .aq-value { font-size: 32px; }
  .forecast-arrow { font-size: 22px; color: var(--muted); text-align: center; }
  .forecast-meta { color: var(--muted); font-size: 11px; text-align: center; margin-top: 4px; }
  .delta-up { color: var(--danger); } .delta-down { color: var(--ok); } .delta-flat { color: var(--muted); }

  .pollutant-select {
    background: var(--card); border: 1px solid var(--border); color: var(--text);
    padding: 6px 10px; border-radius: 8px; font-size: 12px; margin-left: 8px;
  }
  .legend-actual { border-bottom: 2px solid #4ade80; padding-bottom: 1px; }
  .legend-predicted { border-bottom: 2px dashed #ffbe46; padding-bottom: 1px; }
  .not-installed-note {
    color: var(--muted); font-size: 12px; text-align: center; padding: 24px;
    border: 1px dashed var(--border); border-radius: 10px;
  }
</style>
</head>
<body>
  <h1>IAQ Live Dashboard</h1>
  <div class="subtitle">
    <span id="pipeline-badge" class="badge badge-nodata">● checking pipeline...</span>
    <span id="sensor-badge" class="badge badge-nodata" style="display:none;">● sensors</span>
    <span id="forecast-badge" class="badge badge-nodata" style="display:none;">● forecast</span>
    &nbsp; Last updated: <span id="last-updated">—</span>
    &nbsp; <span id="frame-interval" style="color:var(--muted);"></span>
  </div>
  <div id="js-error-banner" class="js-error-banner"></div>
  <div id="alert-banner" class="alert-banner">
    <span class="alert-icon">⚠</span>
    <span id="alert-text"></span>
  </div>

  <div class="video-row">
    <div class="card video-card">
      <h3>Live YOLO inference</h3>
      <img id="live-video" class="video-feed" alt="Live inference stream">
      <div id="video-error" class="video-error"></div>
    </div>

    <div class="status-panel">
      <div class="card people-card">
        <div class="people-card-header">
          <span class="live-dot"></span>
          <h3>People in zone</h3>
        </div>
        <div class="big-number" id="zone-count">—</div>
        <div class="big-number-label">unique people seen this session: <span id="unique-total">—</span></div>
      </div>

      <div class="grid" id="machine-cards" style="grid-template-columns: 1fr;"></div>
    </div>
  </div>

  <div class="card section" id="aq-section">
    <h3>Air Quality — Current Readings</h3>
    <div id="aq-not-installed" class="not-installed-note" style="display:none;">
      No sensor data yet — pollutant sensor ingestion (step 13) isn't installed or hasn't reported yet.
    </div>
    <div id="aq-grid" class="aq-grid"></div>
  </div>

  <div class="card section" id="forecast-section">
    <h3>BiLSTM Forecast — 10 Minutes Ahead</h3>
    <div id="forecast-not-installed" class="not-installed-note" style="display:none;">
      No forecast data yet — the BiLSTM forecast sidecar (step 14) isn't installed or hasn't reported yet.
    </div>
    <div id="forecast-content" style="display:none;">
      <div class="forecast-panel">
        <div class="forecast-side">
          <div class="aq-label">Current (<span id="forecast-current-at">—</span>)</div>
          <div class="aq-value" id="forecast-current-pm25">—</div>
          <div class="forecast-meta">PM2.5, µg/m³</div>
        </div>
        <div class="forecast-arrow">→</div>
        <div class="forecast-side">
          <div class="aq-label">Predicted for <span id="forecast-target-at">—</span></div>
          <div class="aq-value" id="forecast-predicted-pm25">—</div>
          <div class="forecast-meta" id="forecast-delta">—</div>
        </div>
      </div>
      <div class="aq-grid" id="forecast-grid" style="margin-top:16px;"></div>
    </div>
  </div>

  <div class="card section" id="trend-section">
    <h3>Pollutant Trend — Actual vs. Forecast
      <select class="pollutant-select" id="pollutant-select">
        <option value="pm2_5">PM2.5</option>
        <option value="pm1">PM1.0</option>
        <option value="pm10">PM10</option>
        <option value="co2">CO2</option>
        <option value="voc">VOC Index</option>
      </select>
    </h3>
    <div id="trend-not-installed" class="not-installed-note" style="display:none;">
      No trend data yet — install sensor ingestion and the forecast sidecar to populate this chart.
    </div>
    <div id="trend-content" style="display:none;">
      <div class="timeline-legend">
        <span class="legend-item"><span class="legend-actual">Actual reading</span></span>
        <span class="legend-item"><span class="legend-predicted">Forecast (made 10 min earlier)</span></span>
        <span class="legend-note">Forecast line extending past actual data = predictions for the near future</span>
      </div>
      <div class="chart-wrap"><canvas id="trend-chart"></canvas></div>
    </div>
  </div>

  <div class="card section">
    <h3>Time range</h3>
    <div class="controls">
      <button class="preset-btn active" data-mins="1">Last 1 min (live)</button>
      <button class="preset-btn" data-mins="5">Last 5 min</button>
      <button class="preset-btn" data-mins="15">Last 15 min</button>
      <button class="preset-btn" data-mins="60">Last 1 hour</button>
      <button class="preset-btn" data-mins="1440">Today</button>
    </div>
    <div class="range-form">
      <label style="color:var(--muted);font-size:12px;">Custom range:</label>
      <input type="date" id="range-date">
      <input type="time" id="range-start" step="1">
      <span style="color:var(--muted);">to</span>
      <input type="time" id="range-end" step="1">
      <button class="apply-btn" id="apply-range">Apply</button>
    </div>
    <div class="row-count" id="row-count"></div>
  </div>

  <div class="card section">
    <h3>Person count over time</h3>
    <div class="chart-wrap" style="height:220px;"><canvas id="person-chart"></canvas></div>
  </div>

  <div class="card section">
    <h3>Door state timeline</h3>
    <div class="timeline-legend">
      <span class="legend-item"><span class="legend-swatch swatch-closed"></span>Closed</span>
      <span class="legend-item"><span class="legend-swatch swatch-open"></span>Open</span>
      <span class="legend-item legend-note">Per-minute view shades by fraction of the minute open</span>
    </div>
    <div id="timeline-rows"></div>
    <div class="timeline-axis" id="timeline-axis"></div>
  </div>

  <div class="card section">
    <h3>Recent activity</h3>
    <table>
      <thead><tr><th>Time</th><th>M1</th><th>M2</th><th>M3</th><th>Zone Count</th></tr></thead>
      <tbody id="recent-rows"></tbody>
    </table>
  </div>

<script>
// ── Global safety net: if ANYTHING throws anywhere on this page, show it
// on-screen instead of silently freezing the whole dashboard. This is
// what would have told you immediately that Chart.js failed to load,
// instead of just seeing a blank page with no explanation. ──
window.addEventListener('error', (e) => {
  const banner = document.getElementById('js-error-banner');
  banner.style.display = 'block';
  banner.textContent = 'Dashboard script error: ' + e.message +
    ' (open browser dev tools console for details)';
});

// ── Point the live video at the pipeline's integrated stream (same host,
// different port). Isolated with its own error handling so a stream
// outage never affects the rest of the page. ──
try {
  const streamPort = {{ stream_port }};
  const img = document.getElementById('live-video');
  img.src = `http://${window.location.hostname}:${streamPort}/video`;
  img.onerror = () => {
    document.getElementById('video-error').style.display = 'block';
    document.getElementById('video-error').textContent =
      `Cannot reach video stream at port ${streamPort} -- is iaq_pipeline_pi.py running with ENABLE_LIVE_STREAM=True?`;
  };
} catch (err) {
  console.error('Video stream setup failed:', err);
}

let lastSeenTimestamp = null;
let lastSeenAt = Date.now();
let liveMode = true;      // true = auto-refreshing "last 1 min" view

function pillHtml(state) {
  const cls = state === "Open" ? "open" : "closed";
  return `<span class="pill ${cls}">${state}</span>`;
}

function fmtLocalDatetime(date) {
  const p = n => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${p(date.getMonth()+1)}-${p(date.getDate())} ` +
         `${p(date.getHours())}:${p(date.getMinutes())}:${p(date.getSeconds())}`;
}

// ── Person count chart: fixed 0-20 scale, bar chart. Values above 20
// are clamped for display (bar never overflows past the top) but the
// true value still shows in the tooltip, so nothing is silently hidden. ──
const PERSON_COUNT_MAX = 20;
let personChart = null;
try {
  const pctx = document.getElementById('person-chart').getContext('2d');
  personChart = new Chart(pctx, {
    type: 'bar',
    data: { labels: [], datasets: [{
      label: 'Person count', data: [],
      backgroundColor: 'rgba(74,222,128,0.55)',
      borderColor: '#4ade80', borderWidth: 1, borderRadius: 3,
      maxBarThickness: 28,
    }] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const ds = ctx.chart.data.datasets[ctx.datasetIndex];
              const real = (ds.rawValues && ds.rawValues[ctx.dataIndex] !== undefined)
                ? ds.rawValues[ctx.dataIndex] : ctx.parsed.y;
              return `People: ${real}` + (real > PERSON_COUNT_MAX ? ` (off-scale, capped at ${PERSON_COUNT_MAX} on chart)` : '');
            }
          }
        }
      },
      scales: {
        x: { ticks: { color: '#8a8f9b', maxTicksLimit: 10, font: { size: 10 } }, grid: { display: false } },
        y: { min: 0, max: PERSON_COUNT_MAX, ticks: { color: '#8a8f9b', stepSize: 4 },
             grid: { color: '#2a2e38' } },
      }
    }
  });
} catch (err) {
  console.error('Person chart failed to initialize:', err);
}

function updatePersonChart(d) {
  if (!personChart) return;
  personChart.data.labels = d.labels;
  // Plain numbers, clamped for display -- this is the reliable Chart.js
  // format for a category-axis bar chart. The {y, real} object format
  // used previously doesn't plot correctly without an explicit x on
  // each point, which is why bars weren't appearing at all on
  // historical data despite the door timeline (plain canvas, unrelated
  // to Chart.js's data parsing) working fine with the same response.
  personChart.data.datasets[0].data = d.person_count.map(v => Math.min(v, PERSON_COUNT_MAX));
  personChart.data.datasets[0].rawValues = d.person_count;   // true values, for the tooltip only
  personChart.update();
}

// ── Door state timeline: a horizontal colored strip per machine, the
// standard way monitoring dashboards (Grafana's "state timeline" panel,
// for example) show on/off state over time -- far clearer than a
// stepped line for a binary signal. Built with plain canvas so it needs
// no extra charting library. Per-minute data shades continuously between
// closed/open colors by the fraction of that minute the door was open. ──
const TIMELINE_CLOSED = [78, 161, 255];   // matches --closed
const TIMELINE_OPEN = [255, 190, 70];     // matches --open

function timelineColor(value, resolution) {
  const t = resolution === 'minute' ? value : (value >= 0.5 ? 1 : 0);
  const r = Math.round(TIMELINE_CLOSED[0] + (TIMELINE_OPEN[0] - TIMELINE_CLOSED[0]) * t);
  const g = Math.round(TIMELINE_CLOSED[1] + (TIMELINE_OPEN[1] - TIMELINE_CLOSED[1]) * t);
  const b = Math.round(TIMELINE_CLOSED[2] + (TIMELINE_OPEN[2] - TIMELINE_CLOSED[2]) * t);
  return `rgb(${r},${g},${b})`;
}

let currentHistory = null;
const MACHINE_KEYS = { 1: 'm1_open', 2: 'm2_open', 3: 'm3_open' };

function initTimelineRows() {
  const wrap = document.getElementById('timeline-rows');
  wrap.innerHTML = '';
  [1, 2, 3].forEach(m => {
    const row = document.createElement('div');
    row.className = 'timeline-row';
    row.innerHTML = `
      <div class="timeline-label">Machine ${m}</div>
      <div class="timeline-track-wrap">
        <canvas class="timeline-track" id="tl-canvas-${m}" height="30"></canvas>
        <div class="timeline-tooltip" id="tl-tooltip-${m}"></div>
      </div>
      <div class="timeline-current closed" id="tl-current-${m}">—</div>`;
    wrap.appendChild(row);

    const canvas = document.getElementById(`tl-canvas-${m}`);
    const tooltip = document.getElementById(`tl-tooltip-${m}`);
    canvas.addEventListener('mousemove', (e) => {
      if (!currentHistory || currentHistory.labels.length === 0) return;
      const rect = canvas.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
      const idx = Math.min(currentHistory.labels.length - 1, Math.floor(frac * currentHistory.labels.length));
      const val = currentHistory[MACHINE_KEYS[m]][idx];
      const label = currentHistory.resolution === 'minute'
        ? `${currentHistory.labels[idx]} — ${Math.round(val * 100)}% open`
        : `${currentHistory.labels[idx]} — ${val ? 'Open' : 'Closed'}`;
      tooltip.textContent = label;
      tooltip.style.display = 'block';
      tooltip.style.left = `${e.clientX - rect.left}px`;
      tooltip.style.top = '0px';
    });
    canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
  });
}

function drawTimelines(d) {
  currentHistory = d;
  [1, 2, 3].forEach(m => {
    const canvas = document.getElementById(`tl-canvas-${m}`);
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || canvas.parentElement.clientWidth;
    canvas.width = w * dpr; canvas.height = 30 * dpr;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, 30);

    const values = d[MACHINE_KEYS[m]];
    if (!values || values.length === 0) {
      ctx.fillStyle = '#1a1d24'; ctx.fillRect(0, 0, w, 30);
      document.getElementById(`tl-current-${m}`).textContent = '—';
      return;
    }
    const segW = w / values.length;
    values.forEach((v, i) => {
      ctx.fillStyle = timelineColor(v, d.resolution);
      ctx.fillRect(i * segW, 0, segW + 0.5, 30);
    });

    const last = values[values.length - 1];
    const badge = document.getElementById(`tl-current-${m}`);
    const isOpen = d.resolution === 'minute' ? last >= 0.5 : last === 1;
    badge.className = 'timeline-current ' + (isOpen ? 'open' : 'closed');
    badge.textContent = d.resolution === 'minute' ? `${Math.round(last * 100)}%` : (isOpen ? 'Open' : 'Closed');
  });

  const axis = document.getElementById('timeline-axis');
  if (d.labels.length > 1) {
    axis.innerHTML = `<span>${d.labels[0]}</span><span>${d.labels[d.labels.length - 1]}</span>`;
  } else {
    axis.innerHTML = '';
  }
}

initTimelineRows();

function updateHistoryViews(d) {
  updatePersonChart(d);
  drawTimelines(d);
  document.getElementById('row-count').textContent =
    `${d.row_count} data points (${d.resolution === 'minute' ? 'per-minute' : 'per-second'} resolution)`;
}

let lastAqTrendFetchAt = 0;

function loadRange(startStr, endStr) {
  fetch(`/api/history?start=${encodeURIComponent(startStr)}&end=${encodeURIComponent(endStr)}`)
    .then(r => r.json()).then(updateHistoryViews)
    .catch(err => console.error('History fetch failed:', err));

  // In live mode, loadRange() fires every 1s (needed for the person/door
  // view) -- but air-quality data only changes ~once/minute, so refetching
  // it every second would be pure waste. Throttled to once per 4s; a
  // preset/custom-range click always bypasses the throttle (force=true)
  // so switching ranges feels instant, not up-to-4s delayed.
  const now = Date.now();
  if (now - lastAqTrendFetchAt > 4000) {
    lastAqTrendFetchAt = now;
    loadAqTrend(startStr, endStr);
  }
}

// ── Air quality / forecast / alert rendering ─────────────────────────────
const POLLUTANT_LABELS_JS = {pm1:'PM1.0', pm2_5:'PM2.5', pm10:'PM10', co2:'CO₂', voc:'VOC Index'};
const POLLUTANT_UNITS_JS = {pm1:'µg/m³', pm2_5:'µg/m³', pm10:'µg/m³', co2:'ppm', voc:'index'};

function fmtVal(v, decimals) {
  decimals = decimals === undefined ? 1 : decimals;
  return (v === null || v === undefined) ? null : Number(v).toFixed(decimals);
}

function renderAqCard(label, value, unit, classified) {
  const cls = classified && classified.class ? `status-${classified.class}` : '';
  const statusLabel = classified && classified.label ? classified.label : '';
  const fv = fmtVal(value);
  return `
    <div class="aq-card ${cls}">
      <div class="aq-label">${label}</div>
      <div class="aq-value ${fv === null ? 'no-data' : ''}">${fv === null ? 'No data' : fv + '<span class="aq-unit">' + unit + '</span>'}</div>
      <div class="aq-status">${statusLabel}</div>
    </div>`;
}

function setHealthBadge(elemId, health, labelPrefix) {
  const badge = document.getElementById(elemId);
  if (!badge) return;
  if (health.state === 'not_installed') { badge.style.display = 'none'; return; }
  badge.style.display = 'inline-block';
  if (health.state === 'running') {
    badge.className = 'badge badge-ok';
    badge.textContent = `● ${labelPrefix} running`;
  } else if (health.state === 'stale') {
    badge.className = 'badge badge-warn';
    badge.textContent = `● ${labelPrefix} stale (${health.seconds_since_last}s)`;
  } else {
    badge.className = 'badge badge-danger';
    badge.textContent = `● ${labelPrefix} no data`;
  }
}

function updateAirQuality(d) {
  setHealthBadge('sensor-badge', d.sensor_health, 'sensors');
  setHealthBadge('forecast-badge', d.forecast_health, 'forecast');

  const aqNotInstalled = document.getElementById('aq-not-installed');
  const aqGrid = document.getElementById('aq-grid');
  if (!d.available) {
    aqNotInstalled.style.display = 'block';
    aqGrid.style.display = 'none';
  } else {
    aqNotInstalled.style.display = 'none';
    aqGrid.style.display = 'grid';
    let html = '';
    for (const p of ['pm1', 'pm2_5', 'pm10', 'co2', 'voc']) {
      html += renderAqCard(POLLUTANT_LABELS_JS[p], d.current[p], POLLUTANT_UNITS_JS[p], d.current_classified[p]);
    }
    html += renderAqCard('Temperature', d.current.temp, '°C', null);
    html += renderAqCard('Humidity', d.current.hum, '%', null);
    aqGrid.innerHTML = html;
  }

  const forecastNotInstalled = document.getElementById('forecast-not-installed');
  const forecastContent = document.getElementById('forecast-content');
  if (!d.available || !d.predicted) {
    forecastNotInstalled.style.display = 'block';
    forecastContent.style.display = 'none';
  } else {
    forecastNotInstalled.style.display = 'none';
    forecastContent.style.display = 'block';
    document.getElementById('forecast-current-at').textContent = d.current_at || '—';
    document.getElementById('forecast-current-pm25').textContent = fmtVal(d.current.pm2_5) || '—';
    document.getElementById('forecast-target-at').textContent = d.predicted.predicted_for || '—';
    document.getElementById('forecast-predicted-pm25').textContent = fmtVal(d.predicted.pm2_5) || '—';

    const cur = d.current.pm2_5, pred = d.predicted.pm2_5;
    const deltaEl = document.getElementById('forecast-delta');
    if (cur !== null && cur !== undefined && pred !== null && pred !== undefined) {
      const delta = pred - cur;
      const arrow = delta > 0.5 ? '▲' : (delta < -0.5 ? '▼' : '≈');
      const cls = delta > 0.5 ? 'delta-up' : (delta < -0.5 ? 'delta-down' : 'delta-flat');
      deltaEl.innerHTML = `<span class="${cls}">${arrow} ${delta >= 0 ? '+' : ''}${delta.toFixed(1)} µg/m³ vs now</span>`;
    } else {
      deltaEl.textContent = 'PM2.5, µg/m³';
    }

    let gridHtml = '';
    for (const p of ['pm1', 'pm2_5', 'pm10', 'co2', 'voc']) {
      gridHtml += renderAqCard(POLLUTANT_LABELS_JS[p] + ' (predicted)', d.predicted[p], POLLUTANT_UNITS_JS[p], d.predicted_classified[p]);
    }
    document.getElementById('forecast-grid').innerHTML = gridHtml;
  }

  // Alert banner -- ONLY reflects what this view can see (current/predicted
  // readings crossing the "danger" band). This is a VIEW, not a safety
  // system: it doesn't page anyone, doesn't sound an alarm, and doesn't
  // persist an audit trail. Say so explicitly rather than implying more
  // than this dashboard actually does -- see CLAUDE.md's SOS gap note.
  const banner = document.getElementById('alert-banner');
  const alertText = document.getElementById('alert-text');
  if (d.overall_current === 'danger' || d.overall_predicted === 'danger') {
    banner.classList.add('show');
    const parts = [];
    if (d.overall_current === 'danger') parts.push('current reading');
    if (d.overall_predicted === 'danger') parts.push(`predicted reading at ${d.predicted ? d.predicted.predicted_for : ''}`);
    alertText.innerHTML = `<strong>Air quality threshold exceeded</strong> — ${parts.join(' and ')} in the Unhealthy/Very Unhealthy range. ` +
      `This banner is a dashboard VIEW only — no alert has been sent, paged, or logged anywhere. ` +
      `A real SOS system needs an independent hard-threshold alert path; see CLAUDE.md.`;
  } else {
    banner.classList.remove('show');
  }
}

function refreshAirQuality() {
  fetch('/api/air_quality').then(r => r.json()).then(updateAirQuality)
    .catch(err => console.error('Air quality fetch failed:', err));
}

// ── Pollutant trend chart: actual (solid) vs. forecast (dashed), aligned
// on the timestamp each point is FOR (see fetch_air_quality_history's
// docstring) -- the forecast line extending past the actual line's right
// edge is the intended, meaningful behavior (predictions into the near
// future), not a bug. ──
let trendChart = null;
let currentTrendData = null;
try {
  const tctx = document.getElementById('trend-chart').getContext('2d');
  trendChart = new Chart(tctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Actual', data: [], borderColor: '#4ade80', backgroundColor: 'rgba(74,222,128,0.08)',
          borderWidth: 2, pointRadius: 0, tension: 0.25, spanGaps: false, fill: true },
        { label: 'Forecast', data: [], borderColor: '#ffbe46', borderDash: [6, 4],
          borderWidth: 2, pointRadius: 0, tension: 0.25, spanGaps: false, fill: false },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#8a8f9b', maxTicksLimit: 8, font: { size: 10 } }, grid: { display: false } },
        y: { ticks: { color: '#8a8f9b' }, grid: { color: '#2a2e38' } },
      },
    },
  });
} catch (err) {
  console.error('Trend chart failed to initialize:', err);
}

function renderTrendChart() {
  if (!trendChart || !currentTrendData || !currentTrendData.available) return;
  const p = document.getElementById('pollutant-select').value;
  trendChart.data.labels = currentTrendData.labels;
  trendChart.data.datasets[0].data = currentTrendData[`actual_${p}`];
  trendChart.data.datasets[1].data = currentTrendData[`predicted_${p}`];
  trendChart.update();
}

try {
  document.getElementById('pollutant-select').addEventListener('change', renderTrendChart);
} catch (err) {
  console.error('Pollutant selector wiring failed:', err);
}

function loadAqTrend(startStr, endStr) {
  // Extend the queried end by 10 minutes so the forecast's near-future
  // extension is actually visible on the chart rather than clipped at
  // "now" -- the whole point of this chart is seeing predictions ahead
  // of the actual data.
  let endWithBuffer;
  try {
    endWithBuffer = new Date(new Date(endStr.replace(' ', 'T')).getTime() + 10 * 60000);
  } catch (err) {
    endWithBuffer = new Date();
  }
  fetch(`/api/air_quality_history?start=${encodeURIComponent(startStr)}&end=${encodeURIComponent(fmtLocalDatetime(endWithBuffer))}`)
    .then(r => r.json()).then(d => {
      const notInstalled = document.getElementById('trend-not-installed');
      const content = document.getElementById('trend-content');
      if (!d.available || d.row_count === 0) {
        notInstalled.style.display = 'block';
        content.style.display = 'none';
        return;
      }
      notInstalled.style.display = 'none';
      content.style.display = 'block';
      currentTrendData = d;
      renderTrendChart();
    })
    .catch(err => console.error('AQ trend fetch failed:', err));
}

function applyPreset(mins) {
  liveMode = (mins === 1);
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  document.querySelector(`.preset-btn[data-mins="${mins}"]`).classList.add('active');
  const end = new Date();
  // AQ trend is more useful over at least the last hour, even when the
  // person/door "live" preset is a tighter 1-minute window -- a 1-minute
  // pollutant trend chart would show almost nothing.
  const aqMins = Math.max(mins, 60);
  const start = new Date(end.getTime() - mins * 60000);
  const aqStart = new Date(end.getTime() - aqMins * 60000);
  lastAqTrendFetchAt = 0;   // bypass the throttle -- a range change should feel instant
  loadRange(fmtLocalDatetime(start), fmtLocalDatetime(end));
  loadAqTrend(fmtLocalDatetime(aqStart), fmtLocalDatetime(end));
}

document.querySelectorAll('.preset-btn').forEach(btn => {
  btn.addEventListener('click', () => applyPreset(parseInt(btn.dataset.mins)));
});

document.getElementById('apply-range').addEventListener('click', () => {
  liveMode = false;
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  const date = document.getElementById('range-date').value;
  const startT = document.getElementById('range-start').value || '00:00:00';
  const endT = document.getElementById('range-end').value || '23:59:59';
  if (!date) return;
  lastAqTrendFetchAt = 0;
  loadRange(`${date} ${startT}`, `${date} ${endT}`);
});

// Default the date picker to today -- isolated so a failure here can't
// block the status/table refresh loop below either.
try {
  const today = new Date();
  document.getElementById('range-date').value =
    `${today.getFullYear()}-${String(today.getMonth()+1).padStart(2,'0')}-${String(today.getDate()).padStart(2,'0')}`;
} catch (err) {
  console.error('Date picker default failed:', err);
}

function setPipelineBadge(health) {
  const badge = document.getElementById('pipeline-badge');
  const interval = document.getElementById('frame-interval');
  if (health.state === 'running') {
    badge.className = 'badge badge-running';
    badge.textContent = '● pipeline running';
  } else if (health.state === 'stale') {
    badge.className = 'badge badge-stale';
    badge.textContent = `● pipeline stale (${health.seconds_since_last}s since last frame)`;
  } else {
    badge.className = 'badge badge-nodata';
    badge.textContent = '● no data yet -- is the pipeline running?';
  }
  interval.textContent = health.avg_frame_interval_sec
    ? `~${health.avg_frame_interval_sec}s per frame (actual)` : '';
}

function refreshCardsAndTable() {
  fetch('/api/status').then(r => r.json()).then(d => {
    document.getElementById('last-updated').textContent = d.last_updated || '—';
    document.getElementById('zone-count').textContent = d.zone_count;
    document.getElementById('unique-total').textContent = d.unique_total;
    setPipelineBadge(d.pipeline_health);

    const cards = document.getElementById('machine-cards');
    cards.innerHTML = '';
    for (const [m, info] of Object.entries(d.machines)) {
      const cls = info.state === 'Open' ? 'open' : 'closed';
      cards.innerHTML += `
        <div class="card">
          <h3>Machine ${m}</h3>
          <div class="door-state ${cls}">${info.state}</div>
          <div class="door-meta">
            Toggles today: ${info.toggles}<br>
            Time closed: ${info.setup_sec}s &nbsp; Time open: ${info.swap_sec}s
          </div>
        </div>`;
    }

    const rows = document.getElementById('recent-rows');
    rows.innerHTML = d.recent_rows.map(r => `
      <tr>
        <td>${r.Timestamp_ISO8601 ? r.Timestamp_ISO8601.split('T')[1].split('.')[0] : '—'}</td>
        <td>${pillHtml(r.M1_State_Debounced)}</td>
        <td>${pillHtml(r.M2_State_Debounced)}</td>
        <td>${pillHtml(r.M3_State_Debounced)}</td>
        <td>${r.Zone_Count}</td>
      </tr>`).join('');

    if (d.last_updated !== lastSeenTimestamp) {
      lastSeenTimestamp = d.last_updated;
      lastSeenAt = Date.now();
    }
  }).catch(err => {
    console.error('Status fetch failed:', err);
    const badge = document.getElementById('pipeline-badge');
    badge.className = 'badge badge-nodata';
    badge.textContent = '● dashboard cannot reach backend';
  });

  if (liveMode) {
    const end = new Date();
    const start = new Date(end.getTime() - 60000);
    loadRange(fmtLocalDatetime(start), fmtLocalDatetime(end));
  }
}

refreshCardsAndTable();
setInterval(refreshCardsAndTable, 1000);

// Air quality/forecast data only changes ~once/minute -- polling every 5s
// keeps the dashboard feeling live without hammering the DB for no reason.
refreshAirQuality();
setInterval(refreshAirQuality, 5000);

// The AQ trend chart needs a wider default window than the person/door
// "live" view's 1-minute default -- a 1-minute pollutant trend is nearly
// empty. Fetch a sensible 60-minute window explicitly on load, independent
// of whichever vision preset is active (matches the >=60min floor already
// applied inside applyPreset() for every subsequent preset click).
(function initialAqTrend() {
  const end = new Date();
  const start = new Date(end.getTime() - 60 * 60000);
  loadAqTrend(fmtLocalDatetime(start), fmtLocalDatetime(end));
})();
</script>
</body>
</html>
"""


def build_app():
    app = Flask(__name__, static_folder="static", static_url_path="/static")

    @app.route("/")
    def index():
        return render_template_string(INDEX_HTML, stream_port=STREAM_PORT)

    @app.route("/api/status")
    def api_status():
        return jsonify(fetch_status())

    @app.route("/api/history")
    def api_history():
        from flask import request
        start = request.args.get("start")
        end = request.args.get("end")
        resolution = request.args.get("resolution", "auto")
        if not start or not end:
            return jsonify({"error": "start and end query params required"}), 400

        if resolution == "auto":
            # PI: under ~5 minutes, per-second detail is still readable and
            # useful. Beyond that, switch to the pre-aggregated per-minute
            # ct_vectors table -- fewer points, clearer trend, and MUCH
            # cheaper to query over "Today"-scale ranges.
            try:
                from datetime import datetime as _dt
                duration_sec = (_dt.strptime(end, "%Y-%m-%d %H:%M:%S") -
                                 _dt.strptime(start, "%Y-%m-%d %H:%M:%S")).total_seconds()
            except ValueError:
                duration_sec = 999999
            resolution = "second" if duration_sec <= 300 else "minute"

        if resolution == "minute":
            return jsonify(fetch_history_minute(start, end))
        return jsonify(fetch_history(start, end))

    @app.route("/api/air_quality")
    def api_air_quality():
        return jsonify(fetch_air_quality())

    @app.route("/api/air_quality_history")
    def api_air_quality_history():
        from flask import request
        start = request.args.get("start")
        end = request.args.get("end")
        if not start or not end:
            return jsonify({"error": "start and end query params required"}), 400
        return jsonify(fetch_air_quality_history(start, end))

    return app


def main():
    global DB_PATH, STREAM_PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--stream-port", type=int, default=STREAM_PORT,
                     help="Port the pipeline's integrated /video stream is running on")
    args = ap.parse_args()
    DB_PATH = args.db
    STREAM_PORT = args.stream_port

    app = build_app()
    print(f"\nDashboard reading: {DB_PATH}")
    print(f"View at: http://<PI_IP_ADDRESS>:{args.port}")
    print("Find the Pi's IP with: hostname -I")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
