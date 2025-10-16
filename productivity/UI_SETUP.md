# Productivity Tracker UI Setup Guide

## Overview

The Productivity Tracker now includes a modern web-based dashboard that provides three main sections:

1. **Live RTSP Stream Viewer** - Real-time video feed with detection overlays
2. **Live Data Display** - Current metrics with real-time charts and counters
3. **Data Summary & Analytics** - Historical trends and system statistics

## Prerequisites

1. **Python Environment**: Python 3.8 or higher
2. **Dependencies**: Install all requirements from `requirements.txt`
3. **YOLO Models**: At least one YOLO model file (yolov8n.pt, yolov8s.pt, or yolov8m.pt)
4. **RTSP Stream**: A valid RTSP camera URL or video file for testing

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your Stream

Edit `config.yaml` and update the following:

```yaml
input:
  source: "rtsp://your-camera-ip:port/stream" # Replace with your RTSP URL

roi:
  polygon: # Define your region of interest (pixel coordinates)
    - [0, 0]
    - [1920, 0]
    - [1920, 1080]
    - [0, 1080]
```

### 3. Start the Web UI

**Windows:**

```bash
start_ui.bat
```

**Linux/Mac:**

```bash
python start_ui.py
```

### 4. Access the Dashboard

Open your web browser and go to: `http://localhost:5000`

## Dashboard Features

### Live Stream Section

- Real-time RTSP video feed
- Object detection overlays (people, phones)
- ROI polygon visualization
- Connection status indicator
- FPS counter

### Live Data Section

- **People Inside**: Current count of people in ROI
- **Active People**: People moving above threshold speed
- **Loitering**: People staying in one area too long
- **Phone Usage**: People detected with phones
- Real-time line chart showing trends
- Last update timestamp

### Summary Section

- Session averages for all metrics
- Historical trend chart
- System status and uptime
- Processing statistics
- Data point counts

## Controls

- **Start Tracking**: Begin video processing and analysis
- **Stop Tracking**: Stop all processing and clear data
- **Auto-refresh**: Dashboard updates automatically every few seconds

## Configuration Options

### Detection Models

```yaml
models:
  person:
    weights: "yolov8m.pt" # Person detection model
    conf: 0.15 # Confidence threshold
  phone:
    weights: "yolov8m.pt" # Phone detection model
    conf: 0.20
```

### Analytics Settings

```yaml
analytics:
  loiter_seconds: 10.0 # Time before considering someone loitering
  active_speed_px_per_s: 40.0 # Minimum speed to be considered active
  phone_overlap_iou: 0.10 # IoU threshold for phone-person association
```

### UI Display Options

```yaml
output:
  draw:
    show_roi: true # Show ROI polygon on video
    show_tracks: true # Show person tracking boxes
    show_boxes: true # Show detection boxes
    show_labels: true # Show detection labels
    show_counters: true # Show metric counters on video
```

## Troubleshooting

### Common Issues

1. **"No video stream"**

   - Check your RTSP URL in config.yaml
   - Verify camera is accessible from your network
   - Test with VLC or similar player first

2. **"Failed to start tracking"**

   - Ensure YOLO model files exist
   - Check Python dependencies are installed
   - Verify config.yaml syntax is correct

3. **"Connection failed"**

   - Check if port 5000 is available
   - Try accessing via 127.0.0.1:5000 instead of localhost
   - Disable firewall temporarily for testing

4. **Poor performance**
   - Use lighter YOLO model (yolov8n.pt instead of yolov8m.pt)
   - Reduce video resolution in camera settings
   - Increase confidence thresholds to reduce detections

### Performance Tips

- **Model Selection**:

  - `yolov8n.pt`: Fastest, lower accuracy
  - `yolov8s.pt`: Balanced speed/accuracy
  - `yolov8m.pt`: Higher accuracy, slower

- **ROI Configuration**: Smaller ROI areas improve performance
- **Detection Thresholds**: Higher confidence values reduce false positives
- **Frame Rate**: Consider limiting input FPS for better stability

## Advanced Features

### Database Integration

Configure PostgreSQL for data storage:

```yaml
output:
  events:
    summary:
      enabled: true
      mode: pg
      dsn: "postgresql://user:pass@localhost:5432/events"
```

### PPE Compliance

Enable mask detection:

```yaml
branches:
  ppe_compliance: true

ppe_classifiers:
  mask_onnx: "assets/mask_detector.onnx"
```

### Idle Object Detection

Monitor for objects left in restricted areas:

```yaml
branches:
  idle_objects: true

roi:
  empty_areas:
    - name: "Restricted Zone"
      polygon:
        - [100, 100]
        - [500, 100]
        - [500, 400]
        - [100, 400]
```

## Support

For issues or questions:

1. Check the console output for error messages
2. Verify your configuration matches the examples
3. Test with sample video files before using RTSP streams
4. Ensure all dependencies are properly installed
