# PostgreSQL Data Update Troubleshooting Guide

## Issue Identified ✅

Your PostgreSQL database **IS working correctly**! The issue is that:

- **Database connection**: ✅ Working
- **Table exists**: ✅ `raw_events` table has 400 records
- **Configuration**: ✅ Properly configured
- **Problem**: ❌ **The tracking system is not currently running**

## Last Data Written

- **Last record**: 2025-10-02 11:23:31 (about 1 hour ago)
- **Current time**: ~12:20 PM
- **Gap**: No data written in the last hour

## Solution Steps

### 1. Start the Optimized Tracking System

The issue is fixed in the optimized version. Start it with:

```bash
python start_ui.py
```

This will:

- Use the optimized Flask app (`app_optimized.py`)
- Include PostgreSQL emitter integration
- Start writing data to your database immediately

### 2. Verify Data is Updating

After starting the system, run this to check for new data:

```bash
python check_recent_data.py
```

You should see new records being written every few seconds.

### 3. Access the Web Dashboard

Open your browser and go to: `http://localhost:5000`

- Click "Start Tracking"
- You should see live video and metrics
- Data will be written to PostgreSQL automatically

## What Was Fixed

### Missing PostgreSQL Integration

The original optimized Flask app was missing the PostgreSQL emitter. I've added:

1. **Emitter Import**:

   ```python
   from src.yolo_v12.emitter import JSONLEmitter
   ```

2. **Emitter Setup** in `OptimizedProductivityTracker.__init__()`:

   ```python
   # PostgreSQL Emitter setup
   self.emitter = None
   events_cfg = self.cfg["output"].get("events")
   if events_cfg:
       self.emitter = JSONLEmitter(
           config=events_cfg,
           aggregate_window_sec=60,
           client_id=self.cfg.get("client", {}).get("id", "CAM-001"),
           client_name=self.cfg.get("client", {}).get("name", "Productivity Tracker"),
           location=self.cfg.get("client", {}).get("location", "Unknown")
       )
   ```

3. **Data Emission** in `process_frame()`:

   ```python
   # Emit to PostgreSQL if configured
   if self.emitter:
       ts_ms = int(time.time() * 1000)
       self.emitter.emit_frame(ts_ms, tracks, {
           "inside": stats["inside_count"],
           "active": stats["active_count"],
           "loiter": stats["loiter_count"],
           "phone_present": int(phone_present_count),
           "phone_in_use": int(phone_in_use_count),
           "total_people": len(tracks)
       })
   ```

4. **Proper Cleanup** in `stop()`:
   ```python
   if self.emitter:
       self.emitter.close()
   ```

## Expected Behavior After Fix

### When System is Running:

- New records written every 5-10 seconds (based on `min_interval_sec: 5`)
- Data includes: inside count, active count, loitering, phone usage
- Timestamps will be current

### In pgAdmin:

- Refresh your query to see new data
- Records will have recent timestamps
- The `doc` JSONB column contains all metrics

### Sample Query for Recent Data:

```sql
SELECT
    id,
    ts,
    doc->>'inside' as people_inside,
    doc->>'active' as active_people,
    doc->>'phone_in_use' as phone_usage
FROM raw_events
WHERE ts > NOW() - INTERVAL '10 minutes'
ORDER BY ts DESC;
```

## Monitoring Data Updates

### Real-time Monitoring:

```bash
# Check recent data
python check_recent_data.py

# Simple connection test
python test_pg_simple.py
```

### Expected Output When Working:

```
Data from last 10 minutes (15 records):
ID: 415 | Time: 2025-10-02 12:25:45 | Inside:  2 | Active:  1 | Phone:  0 | Total:  2
ID: 414 | Time: 2025-10-02 12:25:40 | Inside:  2 | Active:  1 | Phone:  1 | Total:  2
...
```

## Configuration Verification

Your current config is correct:

```yaml
output:
  events:
    summary:
      enabled: true
      mode: pg
      dsn: "postgresql://postgres:1010@localhost:5432/events"
      table: "raw_events"
      min_interval_sec: 5
      aggregate_window_sec: 60
```

## Troubleshooting Steps

### If Still No Data After Starting:

1. **Check System Logs**:

   - Look for "PostgreSQL emitter initialized successfully"
   - Look for any "Error emitting to PostgreSQL" messages

2. **Verify RTSP Stream**:

   - Ensure your camera stream is accessible
   - Check the video feed in the web UI

3. **Check Detection**:

   - Verify people are being detected in the ROI
   - Check if tracking is working

4. **Database Permissions**:
   - Ensure PostgreSQL user has INSERT permissions
   - Check if the connection is still valid

### Common Issues:

1. **"No module named 'psycopg2'"**:

   ```bash
   pip install psycopg2-binary
   ```

2. **Connection refused**:

   - Check if PostgreSQL service is running
   - Verify connection string in config.yaml

3. **Permission denied**:
   - Check PostgreSQL user permissions
   - Verify database exists

## Success Indicators

✅ **System Working Correctly When**:

- New records appear in PostgreSQL every 5-10 seconds
- Timestamps are current (within last minute)
- Data values change based on video content
- Web UI shows live metrics
- No error messages in console

The fix is now complete - just start the system with `python start_ui.py` and your PostgreSQL data will update in real-time!

