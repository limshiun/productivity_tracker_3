#!/usr/bin/env python3
"""
Startup script for the headless productivity tracker.
This script handles the complete setup and startup process.
"""
import os
import sys
import subprocess
import logging
import time
from pathlib import Path


def setup_logging():
    """Setup logging for the startup script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("startup.log")
        ]
    )


def check_dependencies():
    """Check if all required dependencies are installed."""
    logging.info("Checking dependencies...")

    required_packages = [
        "ultralytics",
        "opencv-python",
        "numpy",
        "pyyaml",
        "shapely",
        "onnxruntime",
        "supervision",
        "psycopg2-binary",
        "schedule"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logging.info(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            logging.warning(f"✗ {package} - missing")

    if missing_packages:
        logging.error(f"Missing packages: {missing_packages}")
        logging.info("Installing missing packages...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages)
            logging.info("Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install dependencies: {e}")
            return False

    return True


def check_postgresql():
    """Check if PostgreSQL is running and accessible."""
    logging.info("Checking PostgreSQL connection...")

    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password="1010",  # Update this to your password
            database="events"
        )
        conn.close()
        logging.info("✓ PostgreSQL connection successful")
        return True
    except Exception as e:
        logging.error(f"✗ PostgreSQL connection failed: {e}")
        logging.info(
            "Please ensure PostgreSQL is running and the database is set up.")
        logging.info("Run 'python setup_database.py' to set up the database.")
        return False


def check_config():
    """Check if configuration file exists and is valid."""
    logging.info("Checking configuration...")

    config_path = Path("config.yaml")
    if not config_path.exists():
        logging.error("✗ config.yaml not found")
        return False

    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check required sections
        required_sections = ["input", "models", "output", "scheduler"]
        for section in required_sections:
            if section not in config:
                logging.error(f"✗ Missing required config section: {section}")
                return False

        logging.info("✓ Configuration file is valid")
        return True

    except Exception as e:
        logging.error(f"✗ Configuration file error: {e}")
        return False


def check_models():
    """Check if required model files exist."""
    logging.info("Checking model files...")

    config_path = Path("config.yaml")
    if not config_path.exists():
        return False

    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        models = config.get("models", {})
        missing_models = []

        for model_type, model_config in models.items():
            if isinstance(model_config, dict) and "weights" in model_config:
                weights_path = model_config["weights"]
                if weights_path and not Path(weights_path).exists():
                    missing_models.append(weights_path)

        if missing_models:
            logging.warning(f"Missing model files: {missing_models}")
            logging.info(
                "The system will download missing models automatically on first run.")
        else:
            logging.info("✓ All model files found")

        return True

    except Exception as e:
        logging.error(f"Error checking models: {e}")
        return False


def start_headless_tracker():
    """Start the headless productivity tracker."""
    logging.info("Starting headless productivity tracker...")

    try:
        # Import and run the headless runner
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from yolo_v12.headless_runner import HeadlessRunner

        runner = HeadlessRunner("config.yaml")
        runner.run()

    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logging.error(f"Error running headless tracker: {e}")
        raise


def main():
    """Main startup function."""
    setup_logging()

    logging.info("=" * 60)
    logging.info("Productivity Tracker - Headless Startup")
    logging.info("=" * 60)

    # Pre-flight checks
    checks = [
        ("Dependencies", check_dependencies),
        ("Configuration", check_config),
        ("Models", check_models),
        ("PostgreSQL", check_postgresql),
    ]

    for check_name, check_func in checks:
        logging.info(f"\n--- {check_name} Check ---")
        if not check_func():
            logging.error(
                f"{check_name} check failed. Please fix the issues and try again.")
            sys.exit(1)

    logging.info("\n" + "=" * 60)
    logging.info("All checks passed! Starting headless tracker...")
    logging.info("=" * 60)

    # Start the tracker
    start_headless_tracker()


if __name__ == "__main__":
    main()
