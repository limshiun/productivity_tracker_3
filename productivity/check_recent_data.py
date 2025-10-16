#!/usr/bin/env python3
"""
Check recent PostgreSQL data to see if it's updating
"""

import psycopg2
from datetime import datetime, timedelta
from src.yolo_v12.config import load_config


def check_recent_data():
    try:
        # Load config
        cfg = load_config("config.yaml")
        events_cfg = cfg["output"].get("events")
        summary_cfg = events_cfg.get("summary")

        dsn = summary_cfg.get("dsn")
        table = summary_cfg.get("table", "raw_events")

        conn = psycopg2.connect(dsn)
        cursor = conn.cursor()

        # Check data from last 10 minutes
        cursor.execute(f"""
            SELECT id, ts, doc->>'inside' as inside_count, doc->>'active' as active_count,
                   doc->>'phone_in_use' as phone_usage, doc->>'total_people' as total_people
            FROM {table} 
            WHERE ts > NOW() - INTERVAL '10 minutes'
            ORDER BY ts DESC 
            LIMIT 10;
        """)

        recent = cursor.fetchall()

        print(f"Data from last 10 minutes ({len(recent)} records):")
        print("-" * 80)

        if recent:
            for r in recent:
                inside = r[2] or "0"
                active = r[3] or "0"
                phone = r[4] or "0"
                total = r[5] or "0"
                print(
                    f"ID: {r[0]:3d} | Time: {r[1]} | Inside: {inside:>2s} | Active: {active:>2s} | Phone: {phone:>2s} | Total: {total:>2s}")
        else:
            print("NO RECENT DATA FOUND!")
            print("This means the system is not currently writing to PostgreSQL.")

        # Check last 5 records regardless of time
        print(f"\nLast 5 records (any time):")
        print("-" * 80)

        cursor.execute(f"""
            SELECT id, ts, doc->>'inside' as inside_count, doc->>'active' as active_count,
                   doc->>'phone_in_use' as phone_usage, doc->>'total_people' as total_people
            FROM {table} 
            ORDER BY ts DESC 
            LIMIT 5;
        """)

        last_records = cursor.fetchall()
        for r in last_records:
            inside = r[2] or "0"
            active = r[3] or "0"
            phone = r[4] or "0"
            total = r[5] or "0"
            time_diff = datetime.now() - r[1].replace(tzinfo=None)
            print(f"ID: {r[0]:3d} | Time: {r[1]} | Inside: {inside:>2s} | Active: {active:>2s} | Phone: {phone:>2s} | Total: {total:>2s} | Age: {time_diff}")

        cursor.close()
        conn.close()

        # Provide diagnosis
        print("\n" + "=" * 80)
        print("DIAGNOSIS:")

        if recent:
            print("✓ PostgreSQL is receiving recent data")
            print("✓ The emitter is working correctly")
            print("→ Check if you're viewing the correct data in pgAdmin")
        else:
            print("✗ No recent data found")
            print("→ The tracking system may not be running")
            print("→ Or the emitter is not configured properly")
            print("→ Try starting the UI with: python start_ui.py")

        return len(recent) > 0

    except Exception as e:
        print(f"ERROR: {e}")
        return False


if __name__ == "__main__":
    check_recent_data()
