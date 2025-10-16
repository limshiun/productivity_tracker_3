#!/usr/bin/env python3
"""
Flask web application for Productivity Tracker UI
Provides three main sections:
1. Live RTSP Stream Viewer
2. Real-time Data Display
3. Data Summary & Analytics
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, Response, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import logging
from collections import deque, defaultdict
import psycopg2
from psycopg2.extras import RealDictCursor
import yaml

# Import your existing modules
from src.yolo_v12.config import load_config
from src.yolo_v12.video_stream import VideoStream
from src.yolo_v12.detector import Detector
from src.yolo_v12.trackers.centroid import CentroidTracker
from src.yolo_v12.analytics import ROIAnalytics
from src.yolo_v12.behaviors import PhoneUsage, PPECompliance, FoodContainers, IdleObjects
from src.yolo_v12.visualize import draw_polygon, draw_tracks, draw_counters

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for tracking system state
current_frame = None
current_stats = {}
frame_lock = threading.Lock()
stats_history = deque(maxlen=1000)  # Keep last 1000 data points
system_running = False
tracker_thread = None

# Database connection
db_config = None


class ProductivityTracker:
    def __init__(self, config_path="config.yaml"):
        self.cfg = load_config(config_path)
        self.setup_logging()
        self.setup_components()
        self.running = False

    def setup_logging(self):
        logging.basicConfig(
            level=getattr(
                logging, self.cfg["logging"]["level"].upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(message)s"
        )

    def setup_components(self):
        # Video stream
        self.vs = VideoStream(
            source=self.cfg["input"]["source"],
            loop_file=self.cfg["input"]["loop_file"],
            read_timeout_sec=self.cfg["input"]["read_timeout_sec"],
            reconnect_delay_sec=self.cfg["input"]["reconnect_delay_sec"]
        )

        # Detector
        self.detector = Detector(self.cfg["models"])

        # Tracker
        self.tracker = None
        if self.cfg["tracking"]["enabled"]:
            fps_cap = self.cfg["analytics"]["fps_assumed"]
            self.tracker = CentroidTracker(
                fps=fps_cap,
                max_lost_secs=self.cfg["tracking"]["max_lost_secs"],
                min_iou_match=self.cfg["tracking"]["min_iou_match"]
            )

        # Analytics
        fps_cap = self.cfg["analytics"]["fps_assumed"]
        self.analytics = ROIAnalytics(
            polygon_pts=self.cfg["roi"]["polygon"],
            fps=fps_cap,
            loiter_seconds=self.cfg["analytics"]["loiter_seconds"],
            active_speed_px_per_s=self.cfg["analytics"]["active_speed_px_per_s"]
        )

        # Behavior analysis
        self.phone_logic = PhoneUsage(
            overlap_iou=self.cfg["analytics"]["phone_overlap_iou"]
        )

        # PPE compliance (if enabled)
        self.ppe_logic = None
        if self.cfg["branches"].get("ppe_compliance", False):
            mask_sess = self.load_onnx_session(
                self.cfg.get("ppe_classifiers", {}).get("mask_onnx")
            )
            sleeve_sess = self.load_onnx_session(
                self.cfg.get("ppe_classifiers", {}).get("sleeves_onnx")
            )
            self.ppe_logic = PPECompliance(mask_sess, sleeve_sess)

    def load_onnx_session(self, path):
        if not path:
            return None
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            return ort.InferenceSession(path, sess_options=opts, providers=['CPUExecutionProvider'])
        except Exception as e:
            logging.warning(f"Failed to load ONNX model {path}: {e}")
            return None

    def start(self):
        self.vs.open()
        self.running = True
        logging.info("Productivity tracker started")

    def stop(self):
        self.running = False
        if self.vs:
            self.vs.release()
        logging.info("Productivity tracker stopped")

    def process_frame(self):
        global current_frame, current_stats, stats_history

        if not self.running:
            return None, None

        ok, frame = self.vs.read()
        if not ok or frame is None:
            return None, None

        # Detection
        dets = self.detector.infer(frame)
        person_dets = dets.get("person", np.zeros((0, 6)))
        phone_dets = dets.get("phone", np.zeros((0, 6)))

        # Tracking
        if self.tracker:
            tracks = self.tracker.update(person_dets)
        else:
            tracks = {i+1: {"bbox": d[:4], "trace": [((d[0]+d[2])/2, (d[1]+d[3])/2)], "attrs": {}}
                      for i, d in enumerate(person_dets)}

        # Analytics
        if self.cfg["branches"]["phone_usage"]:
            self.phone_logic.update(tracks, phone_dets)

        if self.cfg["branches"]["people_behavior"]:
            stats = self.analytics.update(tracks)
        else:
            stats = {"inside_count": 0, "active_count": 0,
                     "loiter_count": 0, "speeds": {}}

        # Calculate phone usage
        phone_present_count = sum(1 for _, t in tracks.items()
                                  if t.get("attrs", {}).get("phone_present", False))
        phone_in_use_count = sum(1 for _, t in tracks.items()
                                 if t.get("attrs", {}).get("phone_in_use", False))

        # Draw visualizations
        vis_frame = frame.copy()
        if self.cfg["output"]["draw"]["show_roi"]:
            draw_polygon(vis_frame, self.cfg["roi"]["polygon"])
        if self.cfg["output"]["draw"]["show_tracks"]:
            draw_tracks(vis_frame, tracks,
                        show_boxes=True,
                        show_labels=True)
        if self.cfg["output"]["draw"]["show_counters"]:
            draw_counters(vis_frame, {
                "inside": stats["inside_count"],
                "active": stats["active_count"],
                "loiter": stats["loiter_count"],
                "phone_present": int(phone_present_count),
                "phone_in_use": int(phone_in_use_count),
            })

        # Update global state
        with frame_lock:
            current_frame = vis_frame
            current_stats = {
                "timestamp": datetime.now().isoformat(),
                "inside_count": stats["inside_count"],
                "active_count": stats["active_count"],
                "loiter_count": stats["loiter_count"],
                "phone_present": int(phone_present_count),
                "phone_in_use": int(phone_in_use_count),
                "total_people": len(tracks)
            }
            stats_history.append(current_stats.copy())

        return vis_frame, current_stats


# Global tracker instance
tracker = None


def tracking_worker():
    """Background thread for processing video frames"""
    global tracker, system_running

    while system_running:
        try:
            if tracker:
                frame, stats = tracker.process_frame()
                if stats:
                    # Emit real-time data to connected clients
                    socketio.emit('live_data', stats)
                time.sleep(0.1)  # ~10 FPS processing
        except Exception as e:
            logging.error(f"Error in tracking worker: {e}")
            time.sleep(1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    global tracker, system_running, tracker_thread

    try:
        if not system_running:
            tracker = ProductivityTracker()
            tracker.start()
            system_running = True

            # Start background processing thread
            tracker_thread = threading.Thread(
                target=tracking_worker, daemon=True)
            tracker_thread.start()

        return jsonify({"status": "success", "message": "Tracking started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    global tracker, system_running

    try:
        system_running = False
        if tracker:
            tracker.stop()
            tracker = None
        return jsonify({"status": "success", "message": "Tracking stopped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        global current_frame
        while True:
            with frame_lock:
                if current_frame is not None:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', current_frame,
                                               [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)  # ~10 FPS

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/current_stats')
def get_current_stats():
    """Get current real-time statistics"""
    global current_stats
    return jsonify(current_stats)


@app.route('/api/stats_history')
def get_stats_history():
    """Get historical statistics"""
    global stats_history
    # Return last N points based on query parameter
    limit = request.args.get('limit', 100, type=int)
    return jsonify(list(stats_history)[-limit:])


@app.route('/api/summary')
def get_summary():
    """Get summary statistics for different time periods"""
    try:
        # Calculate summary from recent data
        if not stats_history:
            return jsonify({
                "last_hour": {},
                "last_day": {},
                "current_session": {}
            })

        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        # Filter data by time periods
        recent_data = list(stats_history)

        def calculate_averages(data):
            if not data:
                return {}

            total_points = len(data)
            avg_inside = sum(d.get('inside_count', 0)
                             for d in data) / total_points
            avg_active = sum(d.get('active_count', 0)
                             for d in data) / total_points
            avg_loiter = sum(d.get('loiter_count', 0)
                             for d in data) / total_points
            avg_phone_use = sum(d.get('phone_in_use', 0)
                                for d in data) / total_points

            return {
                "avg_people_inside": round(avg_inside, 1),
                "avg_active_people": round(avg_active, 1),
                "avg_loitering": round(avg_loiter, 1),
                "avg_phone_usage": round(avg_phone_use, 1),
                "total_data_points": total_points
            }

        summary = {
            "current_session": calculate_averages(recent_data),
            "system_status": "running" if system_running else "stopped",
            "last_update": current_stats.get("timestamp", ""),
            "total_data_points": len(stats_history)
        }

        return jsonify(summary)

    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return jsonify({"error": str(e)})


@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to Productivity Tracker'})


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    # Ensure templates and static directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)

    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
