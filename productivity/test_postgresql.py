#!/usr/bin/env python3
"""
PostgreSQL Connection Test Script
Tests the database connection and emitter functionality
"""

import sys
import logging
import time
from datetime import datetime
from src.yolo_v12.config import load_config
from src.yolo_v12.emitter import JSONLEmitter


def test_postgresql_connection():
    """Test PostgreSQL connection and data insertion"""

    print("Testing PostgreSQL Connection...")
    print("=" * 50)

    try:
        # Load configuration
        cfg = load_config("config.yaml")
        events_cfg = cfg["output"].get("events")

        if not events_cfg:
            print("ERROR: No events configuration found in config.yaml")
            return False

        summary_cfg = events_cfg.get("summary")
        if not summary_cfg or summary_cfg.get("mode") != "pg":
            print("ERROR: PostgreSQL mode not enabled in config.yaml")
            print("   Current config:", summary_cfg)
            return False

        print("SUCCESS: Configuration loaded successfully")
        print(f"   DSN: {summary_cfg.get('dsn', 'Not specified')}")
        print(f"   Table: {summary_cfg.get('table', 'raw_events')}")
        print(f"   Enabled: {summary_cfg.get('enabled', False)}")

        # Test database connection
        print("\n🔗 Testing database connection...")

        try:
            import psycopg2
            dsn = summary_cfg.get("dsn")
            if not dsn:
                print("ERROR: No DSN specified in configuration")
                return False

            # Test direct connection
            conn = psycopg2.connect(dsn)
            conn.autocommit = True
            cursor = conn.cursor()

            # Test basic query
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"SUCCESS: Connected to PostgreSQL: {version[:50]}...")

            # Check if table exists
            table_name = summary_cfg.get("table", "raw_events")
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table_name,))

            table_exists = cursor.fetchone()[0]
            print(f"SUCCESS: Table '{table_name}' exists: {table_exists}")

            # Count existing records
            if table_exists:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                count = cursor.fetchone()[0]
                print(f"SUCCESS: Current record count: {count}")

                # Show recent records
                cursor.execute(f"""
                    SELECT id, ts, doc->>'inside' as inside_count, doc->>'active' as active_count 
                    FROM {table_name} 
                    ORDER BY ts DESC 
                    LIMIT 5;
                """)
                recent = cursor.fetchall()
                if recent:
                    print("SUCCESS: Recent records:")
                    for record in recent:
                        print(
                            f"   ID: {record[0]}, Time: {record[1]}, Inside: {record[2]}, Active: {record[3]}")
                else:
                    print("WARNING: No records found in table")

            cursor.close()
            conn.close()

        except ImportError:
            print("❌ psycopg2 not installed. Install with: pip install psycopg2-binary")
            return False
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

        # Test emitter functionality
        print("\n📤 Testing emitter functionality...")

        try:
            emitter = JSONLEmitter(
                config=events_cfg,
                aggregate_window_sec=60,
                client_id=cfg.get("client", {}).get("id", "TEST-CAM"),
                client_name=cfg.get("client", {}).get("name", "Test Camera"),
                location=cfg.get("client", {}).get("location", "Test Location")
            )
            print("✅ Emitter initialized successfully")

            # Test data emission
            test_tracks = {
                1: {"bbox": [100, 100, 200, 200], "trace": [(150, 150)], "attrs": {"phone_present": True}},
                2: {"bbox": [300, 300, 400, 400], "trace": [(350, 350)], "attrs": {"phone_in_use": False}}
            }

            test_counts = {
                "inside": 2,
                "active": 1,
                "loiter": 0,
                "phone_present": 1,
                "phone_in_use": 0,
                "total_people": 2
            }

            ts_ms = int(time.time() * 1000)
            emitter.emit_frame(ts_ms, test_tracks, test_counts)
            print("✅ Test data emitted successfully")

            # Wait a moment and check if data was inserted
            time.sleep(2)

            # Verify insertion
            conn = psycopg2.connect(dsn)
            conn.autocommit = True
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE ts > NOW() - INTERVAL '10 seconds';")
            recent_count = cursor.fetchone()[0]
            print(f"✅ Recent insertions (last 10 seconds): {recent_count}")

            cursor.close()
            conn.close()
            emitter.close()

            return True

        except Exception as e:
            print(f"❌ Emitter test failed: {e}")
            return False

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def show_configuration_help():
    """Show help for configuring PostgreSQL"""

    print("\n" + "=" * 60)
    print("🛠️  POSTGRESQL CONFIGURATION HELP")
    print("=" * 60)

    print("\n1. Ensure PostgreSQL is running:")
    print("   - Check if PostgreSQL service is active")
    print("   - Default port: 5432")

    print("\n2. Create database (if needed):")
    print("   CREATE DATABASE events;")

    print("\n3. Update config.yaml:")
    print("""
   output:
     events:
       summary:
         enabled: true
         mode: pg
         dsn: "postgresql://username:password@localhost:5432/events"
         table: "raw_events"
         min_interval_sec: 5
         aggregate_window_sec: 60
   """)

    print("\n4. Install required packages:")
    print("   pip install psycopg2-binary")

    print("\n5. Test connection:")
    print("   python test_postgresql.py")


def main():
    """Main test function"""

    print("PostgreSQL Integration Test")
    print("=" * 50)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    success = test_postgresql_connection()

    if success:
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ PostgreSQL integration is working correctly")
        print("✅ Data should be updating in your database")
        print("\nYou can now start the UI with: python start_ui.py")
    else:
        print("\n" + "=" * 50)
        print("❌ TESTS FAILED!")
        print("🔧 PostgreSQL integration needs configuration")
        show_configuration_help()

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
