# Productivity Tracker - Vision Analytics Dashboard

A comprehensive productivity tracking system with real-time RTSP stream analysis and web-based dashboard. The system monitors people behavior, phone usage, PPE compliance, and provides detailed analytics through an intuitive web interface.

## Features

- **Live RTSP Stream Viewing**: Real-time video feed with object detection overlays
- **Real-time Analytics**: Live tracking of people count, activity levels, phone usage, and loitering
- **Interactive Dashboard**: Modern web interface with three main sections:
  1. Live RTSP stream viewer with detection overlays
  2. Real-time data display with charts and metrics
  3. Summary analytics with historical trends
- **PPE Compliance**: Mask and safety equipment detection (if models provided)
- **Behavior Analysis**: Activity tracking within designated ROI polygons
- **Data Export**: PostgreSQL integration for data storage and analysis

## Quick Start

### 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configuration

1. Edit `config.yaml`:
   - Set your RTSP stream URL in `input.source`
   - Adjust `roi.polygon` coordinates (image-space pixels)
   - Configure detection models and thresholds

### 3. Run the Web Dashboard

**Option A: Using the startup script (Recommended)**

```bash
python start_ui.py
```

**Option B: Using the batch file (Windows)**

```bash
start_ui.bat
```

**Option C: Direct Flask run**

```bash
python app.py
```

### 4. Access the Dashboard

Open your web browser and navigate to: `http://localhost:5000`

The dashboard provides:

- **Live Stream**: Real-time RTSP feed with detection overlays
- **Live Data**: Current metrics and real-time charts
- **Summary**: Historical analytics and system status

## Command Line Interface (Legacy)

You can still run the system in headless mode:

```bash
python -m src.main --config config.yaml
```

## Empty areas (idle objects) configuration

Add one or more polygons under roi.empty_areas to watch for non-human objects lingering in places that should be empty. Coordinates are pixel positions in the video frame:

```yaml
roi:
  polygon:
    - [0, 0]
    - [1200, 0]
    - [1200, 799]
    - [0, 799]
  empty_areas:
    - name: "Aisle-Left"
      polygon:
        - [100, 100]
        - [500, 100]
        - [500, 400]
        - [100, 400]
    - name: "Dock-1"
      polygon:
        - [700, 200]
        - [1100, 220]
        - [1080, 600]
        - [680, 580]
```

Tips:

- Use at least 3 points per polygon; 4 (rectangle) is common.
- Values must be numbers; strings or malformed lists will be ignored.
- You can also use the key points instead of polygon if you prefer the name.
- If a malformed area is provided, the app will skip it and continue.

## Notes

- The default detector uses the Ultralytics YOLO family (change weights in `config.yaml`).
- The default tracker is a lightweight CentroidTracker. You can extend with StrongSORT/ByteTrack later via `yolo_v12/trackers/*`.
- All counts are **best-effort heuristics** and meant as a baseline for your environment.
