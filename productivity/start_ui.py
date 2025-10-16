#!/usr/bin/env python3
"""
Startup script for the Productivity Tracker Web UI
"""

import os
import sys
import logging
from pathlib import Path


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Check if required files exist
    required_files = [
        "config.yaml",
        "app.py",
        "templates/index.html",
        "static/css/style.css",
        "static/js/app.js"
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        logging.error("Missing required files:")
        for file_path in missing_files:
            logging.error(f"  - {file_path}")
        sys.exit(1)

    # Check if YOLO model files exist
    model_files = ["yolov8m.pt", "yolov8n.pt", "yolov8s.pt"]
    available_models = [f for f in model_files if Path(f).exists()]

    if not available_models:
        logging.warning(
            "No YOLO model files found. Please ensure you have at least one of:")
        for model in model_files:
            logging.warning(f"  - {model}")
    else:
        logging.info(f"Found YOLO models: {', '.join(available_models)}")

    # Start the Flask application
    logging.info("Starting Productivity Tracker Web UI...")
    logging.info("Access the dashboard at: http://localhost:5000")
    logging.info("Press Ctrl+C to stop the server")

    try:
        # Import and run the optimized Flask app
        from app_optimized import app, socketio
        socketio.run(app, debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
