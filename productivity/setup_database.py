#!/usr/bin/env python3
"""
Database setup script for the productivity tracker.
Creates the necessary PostgreSQL database and tables.
"""
import psycopg2
import psycopg2.extensions
import sys
import logging
from typing import Optional


def setup_logging():
    """Setup logging for the database setup script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )


def create_database(host: str, port: int, user: str, password: str,
                    database: str = "postgres") -> bool:
    """
    Create the events database if it doesn't exist.

    Args:
        host: PostgreSQL host
        port: PostgreSQL port
        user: PostgreSQL user
        password: PostgreSQL password
        database: Database to connect to (default: postgres)

    Returns:
        True if database was created or already exists, False otherwise
    """
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        conn.autocommit = True
        cur = conn.cursor()

        # Check if events database exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = 'events'")
        if cur.fetchone():
            logging.info("Database 'events' already exists")
        else:
            # Create events database
            cur.execute("CREATE DATABASE events")
            logging.info("Database 'events' created successfully")

        cur.close()
        conn.close()
        return True

    except Exception as e:
        logging.error(f"Error creating database: {e}")
        return False


def create_tables(host: str, port: int, user: str, password: str,
                  database: str = "events") -> bool:
    """
    Create the necessary tables in the events database.

    Args:
        host: PostgreSQL host
        port: PostgreSQL port
        user: PostgreSQL user
        password: PostgreSQL password
        database: Database name (default: events)

    Returns:
        True if tables were created successfully, False otherwise
    """
    try:
        # Connect to events database
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        conn.autocommit = True
        cur = conn.cursor()

        # Create raw_events table
        create_raw_events_table = """
        CREATE TABLE IF NOT EXISTS raw_events (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMPTZ DEFAULT now(),
            doc JSONB NOT NULL
        )
        """
        cur.execute(create_raw_events_table)
        logging.info("Table 'raw_events' created/verified")

        # Create indexes for better performance
        create_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_raw_events_ts ON raw_events(ts)",
            "CREATE INDEX IF NOT EXISTS idx_raw_events_doc_type ON raw_events USING GIN ((doc->>'type'))",
            "CREATE INDEX IF NOT EXISTS idx_raw_events_doc_client_id ON raw_events USING GIN ((doc->>'client_id'))"
        ]

        for index_sql in create_indexes:
            cur.execute(index_sql)

        logging.info("Indexes created/verified")

        # Create a summary view for easier querying
        create_summary_view = """
        CREATE OR REPLACE VIEW daily_summary AS
        SELECT 
            DATE(ts) as date,
            (doc->>'client_id') as client_id,
            (doc->>'client_name') as client_name,
            (doc->>'location') as location,
            AVG((doc->>'inside')::int) as avg_inside,
            AVG((doc->>'active')::int) as avg_active,
            AVG((doc->>'loiter')::int) as avg_loiter,
            AVG((doc->>'phone_present')::int) as avg_phone_present,
            AVG((doc->>'phone_in_use')::int) as avg_phone_in_use,
            AVG((doc->>'containers_moved')::int) as avg_containers_moved,
            AVG((doc->>'3idle_objects')::int) as avg_idle_objects,
            COUNT(*) as total_events
        FROM raw_events 
        WHERE doc->>'type' = 'summary'
        GROUP BY DATE(ts), (doc->>'client_id'), (doc->>'client_name'), (doc->>'location')
        ORDER BY date DESC
        """
        cur.execute(create_summary_view)
        logging.info("Summary view created/updated")

        cur.close()
        conn.close()
        return True

    except Exception as e:
        logging.error(f"Error creating tables: {e}")
        return False


def test_connection(host: str, port: int, user: str, password: str,
                    database: str = "events") -> bool:
    """
    Test the database connection.

    Args:
        host: PostgreSQL host
        port: PostgreSQL port
        user: PostgreSQL user
        password: PostgreSQL password
        database: Database name (default: events)

    Returns:
        True if connection successful, False otherwise
    """
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        cur = conn.cursor()
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        logging.info(f"Connected to PostgreSQL: {version}")

        cur.close()
        conn.close()
        return True

    except Exception as e:
        logging.error(f"Connection test failed: {e}")
        return False


def main():
    """Main setup function."""
    setup_logging()

    # Database configuration - update these values as needed
    DB_CONFIG = {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "1010",  # Update this to your PostgreSQL password
        "database": "events"
    }

    logging.info("Setting up PostgreSQL database for productivity tracker...")

    # Step 1: Create database
    logging.info("Step 1: Creating database...")
    if not create_database(**DB_CONFIG):
        logging.error("Failed to create database. Exiting.")
        sys.exit(1)

    # Step 2: Create tables
    logging.info("Step 2: Creating tables...")
    if not create_tables(**DB_CONFIG):
        logging.error("Failed to create tables. Exiting.")
        sys.exit(1)

    # Step 3: Test connection
    logging.info("Step 3: Testing connection...")
    if not test_connection(**DB_CONFIG):
        logging.error("Connection test failed. Exiting.")
        sys.exit(1)

    logging.info("Database setup completed successfully!")
    logging.info(
        "You can now run the productivity tracker with headless mode.")

    # Print connection string for reference
    dsn = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    logging.info(f"Connection string: {dsn}")


if __name__ == "__main__":
    main()
