#!/usr/bin/env python3
"""
Simple PostgreSQL Connection Test
"""

import sys
import time
from src.yolo_v12.config import load_config


def test_connection():
    try:
        # Load config
        cfg = load_config("config.yaml")
        events_cfg = cfg["output"].get("events")
        summary_cfg = events_cfg.get("summary")

        print("Configuration:")
        print(f"  DSN: {summary_cfg.get('dsn')}")
        print(f"  Table: {summary_cfg.get('table')}")
        print(f"  Enabled: {summary_cfg.get('enabled')}")

        # Test database
        import psycopg2
        dsn = summary_cfg.get("dsn")
        conn = psycopg2.connect(dsn)
        cursor = conn.cursor()

        # Check table
        table = summary_cfg.get("table", "raw_events")
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"Records in {table}: {count}")

        # Check recent records
        cursor.execute(f"""
            SELECT id, ts, doc->>'inside' as inside_count 
            FROM {table} 
            ORDER BY ts DESC 
            LIMIT 3;
        """)
        recent = cursor.fetchall()
        print("Recent records:")
        for r in recent:
            print(f"  ID: {r[0]}, Time: {r[1]}, Inside: {r[2]}")

        cursor.close()
        conn.close()

        print("SUCCESS: Database connection working!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


if __name__ == "__main__":
    test_connection()

