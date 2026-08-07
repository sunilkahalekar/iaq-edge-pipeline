"""
query_db.py — quick sanity check on the SQLite output.

Usage:
    python query_db.py --db /home/pi/iaq_data/iaq.db --table ct_vectors --limit 10
"""

import argparse
import sqlite3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", default="ct_vectors",
                     choices=["ct_vectors", "per_second_analytics", "tracking_telemetry"])
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT * FROM {args.table} ORDER BY id DESC LIMIT ?", (args.limit,)
    ).fetchall()

    if not rows:
        print("No rows yet.")
        return

    cols = rows[0].keys()
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r[c]) for c in cols))

    conn.close()


if __name__ == "__main__":
    main()
