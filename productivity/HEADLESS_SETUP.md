# Headless Productivity Tracker Setup

This guide will help you set up the productivity tracker to run 24/7 in headless mode with automatic daily scheduling (7am-5pm) and PostgreSQL data storage.

## Prerequisites

1. **Python 3.8+** installed on your system
2. **PostgreSQL** installed and running
3. **Camera/RTSP stream** accessible
4. **Windows/Linux** system for 24/7 operation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up PostgreSQL Database

```bash
python setup_database.py
```

This will:

- Create the `events` database
- Create the `raw_events` table
- Set up indexes for performance
- Create a summary view for easy querying

### 3. Configure the System

Edit `config.yaml` to match your setup:

```yaml
# Update your camera source
input:
  source: "rtsp://admin:password@192.168.1.100:554/stream"

# Update PostgreSQL connection
output:
  events:
    summary:
      dsn: "postgresql://postgres:YOUR_PASSWORD@localhost:5432/events"

# Scheduler settings (7am-5pm by default)
scheduler:
  start_hour: 7
  end_hour: 17
```

### 4. Start the Headless System

```bash
python start_headless.py
```

## Configuration Details

### Scheduler Configuration

The system automatically captures data daily between specified hours:

```yaml
scheduler:
  start_hour: 7 # 7 AM
  start_minute: 0
  end_hour: 17 # 5 PM
  end_minute: 0
  timezone_offset: 0 # Local timezone
```

### Database Configuration

The system stores data in PostgreSQL with the following structure:

- **Table**: `raw_events`
- **Columns**:
  - `id`: Auto-incrementing primary key
  - `ts`: Timestamp (automatically set)
  - `doc`: JSONB document containing all tracking data

### Data Structure

Each event document contains:

```json
{
  "type": "summary",
  "timestamp": 1640995200000,
  "client_id": "CAM-001",
  "client_name": "My Camera",
  "location": "Warehouse A",
  "inside": 3,
  "active": 2,
  "loiter": 1,
  "phone_present": 1,
  "phone_in_use": 0,
  "containers_moved": 0,
  "idle_objects": 0,
  "idle_max_dwell_sec": 0.0
}
```

## Monitoring and Logs

### Log Files

- `productivity_tracker.log`: Main application logs
- `startup.log`: Startup process logs

### Database Queries

View daily summaries:

```sql
SELECT * FROM daily_summary
WHERE date = CURRENT_DATE
ORDER BY date DESC;
```

View recent events:

```sql
SELECT ts, doc->>'inside' as inside_count, doc->>'active' as active_count
FROM raw_events
WHERE doc->>'type' = 'summary'
ORDER BY ts DESC
LIMIT 100;
```

## System Requirements

### Minimum Requirements

- **CPU**: 4 cores, 2.0 GHz
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **Network**: Stable connection to camera

### Recommended Requirements

- **CPU**: 8 cores, 3.0 GHz
- **RAM**: 16 GB
- **Storage**: 50 GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster processing)

## Troubleshooting

### Common Issues

1. **PostgreSQL Connection Failed**

   - Ensure PostgreSQL is running
   - Check username/password in config
   - Verify database exists

2. **Camera Connection Failed**

   - Test RTSP URL with VLC player
   - Check network connectivity
   - Verify camera credentials

3. **High CPU Usage**

   - Reduce model confidence thresholds
   - Disable unused branches in config
   - Use smaller YOLO model (yolov8n.pt)

4. **Memory Issues**
   - Increase system RAM
   - Reduce batch size in models
   - Restart system periodically

### Performance Optimization

1. **For Lower-End Systems**:

   ```yaml
   models:
     person:
       weights: "yolov8n.pt" # Use nano model
       conf: 0.25 # Higher confidence threshold
   ```

2. **Disable Unused Features**:
   ```yaml
   branches:
     phone_usage: true
     ppe_compliance: false # Disable if not needed
     people_behavior: true
     food_containers: false # Disable if not needed
     idle_objects: false # Disable if not needed
   ```

## Running as a Service

### Windows Service

Create a batch file `start_service.bat`:

```batch
@echo off
cd /d "C:\path\to\productivity_tracker"
python start_headless.py
```

Use Task Scheduler to run this at system startup.

### Linux Service

Create `/etc/systemd/system/productivity-tracker.service`:

```ini
[Unit]
Description=Productivity Tracker
After=network.target postgresql.service

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/productivity_tracker
ExecStart=/usr/bin/python3 start_headless.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable productivity-tracker
sudo systemctl start productivity-tracker
```

## Data Backup

### Automated Backup Script

Create `backup_database.py`:

```python
import psycopg2
import subprocess
from datetime import datetime

# Backup the events database
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_file = f"events_backup_{timestamp}.sql"

subprocess.run([
    "pg_dump",
    "-h", "localhost",
    "-U", "postgres",
    "-d", "events",
    "-f", backup_file
])
```

Schedule this script to run daily using cron (Linux) or Task Scheduler (Windows).

## Support

For issues and questions:

1. Check the log files for error messages
2. Verify all dependencies are installed
3. Test database and camera connections separately
4. Review configuration settings

The system is designed to be robust and self-recovering, automatically restarting capture during scheduled hours even after system reboots.
