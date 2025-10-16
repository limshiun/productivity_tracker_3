#!/usr/bin/env python3
"""
Optimized Flask web application for Productivity Tracker UI
Provides three main sections with improved performance:
1. Live RTSP Stream Viewer (optimized streaming)
2. Real-time Data Display (efficient updates)
3. Data Summary & Analytics (cached data)
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
import yaml
from queue import Queue
import base64

# Import your existing modules
from src.yolo_v12.config import load_config
from src.yolo_v12.video_stream import VideoStream
from src.yolo_v12.detector import Detector
from src.yolo_v12.trackers.centroid import CentroidTracker
from src.yolo_v12.analytics import ROIAnalytics
from src.yolo_v12.behaviors import PhoneUsage, PPECompliance, FoodContainers, IdleObjects
from src.yolo_v12.visualize import draw_polygon, draw_tracks, draw_counters
from src.yolo_v12.emitter import JSONLEmitter

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables for tracking system state
current_frame = None
current_stats = {}
frame_lock = threading.Lock()
stats_history = deque(maxlen=500)  # Reduced for better performance
system_running = False
tracker_thread = None
frame_queue = Queue(maxsize=2)  # Small queue to prevent lag

# Performance optimization variables
frame_skip_counter = 0
FRAME_SKIP = 2  # Process every 3rd frame for better performance
TARGET_WIDTH = 640  # Resize frames for faster processing
last_emit_time = 0
EMIT_INTERVAL = 0.5  # Emit data every 500ms max


class OptimizedProductivityTracker:
    def __init__(self, config_path="config.yaml"):
        self.cfg = load_config(config_path)
        self.setup_logging()
        self.setup_components()
        self.running = False
        self.frame_count = 0
        self.last_process_time = time.time()

    def setup_logging(self):
        logging.basicConfig(
            level=getattr(
                logging, self.cfg["logging"]["level"].upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(message)s"
        )

    def setup_components(self):
        # Video stream with optimizations
        self.vs = VideoStream(
            source=self.cfg["input"]["source"],
            loop_file=self.cfg["input"]["loop_file"],
            read_timeout_sec=self.cfg["input"]["read_timeout_sec"],
            reconnect_delay_sec=self.cfg["input"]["reconnect_delay_sec"]
        )

        # Detector
        self.detector = Detector(self.cfg["models"])

        # Tracker with optimized settings
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
            if mask_sess or sleeve_sess:
                self.ppe_logic = PPECompliance(mask_sess, sleeve_sess)

        # PostgreSQL Emitter setup
        self.emitter = None
        events_cfg = self.cfg["output"].get("events")
        if events_cfg:
            try:
                self.emitter = JSONLEmitter(
                    config=events_cfg,
                    aggregate_window_sec=self.cfg["output"].get(
                        "jsonl", {}).get("aggregate_window_sec", 60),
                    client_id=self.cfg.get("client", {}).get("id", "CAM-001"),
                    client_name=self.cfg.get("client", {}).get(
                        "name", "Productivity Tracker"),
                    location=self.cfg.get("client", {}).get(
                        "location", "Unknown")
                )
                logging.info("PostgreSQL emitter initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize PostgreSQL emitter: {e}")
                self.emitter = None
        elif self.cfg["output"].get("jsonl", {}).get("enabled", False):
            try:
                self.emitter = JSONLEmitter(
                    self.cfg["output"]["jsonl"]["path"],
                    aggregate_window_sec=self.cfg["output"]["jsonl"]["aggregate_window_sec"]
                )
                logging.info("File emitter initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize file emitter: {e}")
                self.emitter = None

    def load_onnx_session(self, path):
        if not path or not os.path.exists(path):
            return None
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return ort.InferenceSession(path, sess_options=opts, providers=['CPUExecutionProvider'])
        except Exception as e:
            logging.warning(f"Failed to load ONNX model {path}: {e}")
            return None

    def start(self):
        self.vs.open()
        self.running = True
        self.frame_count = 0
        logging.info("Optimized productivity tracker started")

    def stop(self):
        self.running = False
        if self.vs:
            self.vs.release()
        if self.emitter:
            try:
                self.emitter.close()
                logging.info("PostgreSQL emitter closed")
            except Exception as e:
                logging.error(f"Error closing emitter: {e}")
        logging.info("Optimized productivity tracker stopped")

    def resize_frame(self, frame, target_width=TARGET_WIDTH):
        """Resize frame while maintaining aspect ratio"""
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w <= target_width:
            return frame

        # Calculate new dimensions
        ratio = target_width / w
        new_h = int(h * ratio)

        # Resize frame
        resized = cv2.resize(frame, (target_width, new_h),
                             interpolation=cv2.INTER_LINEAR)
        return resized

    def process_frame(self):
        global current_frame, current_stats, stats_history, frame_skip_counter, last_emit_time

        if not self.running:
            return None, None

        ok, frame = self.vs.read()
        if not ok or frame is None:
            return None, None

        self.frame_count += 1

        # Frame skipping for performance
        frame_skip_counter += 1
        if frame_skip_counter <= FRAME_SKIP:
            # Still update the display frame even if we skip processing
            with frame_lock:
                current_frame = frame
            return frame, None

        frame_skip_counter = 0

        # Resize frame for faster processing
        process_frame = self.resize_frame(frame)
        if process_frame is None:
            return None, None

        # Calculate scale factor for coordinate conversion
        orig_h, orig_w = frame.shape[:2]
        proc_h, proc_w = process_frame.shape[:2]
        scale_x = orig_w / proc_w
        scale_y = orig_h / proc_h

        # Detection on resized frame
        start_time = time.time()
        dets = self.detector.infer(process_frame)

        # Scale detection coordinates back to original frame size
        person_dets = dets.get("person", np.zeros((0, 6)))
        phone_dets = dets.get("phone", np.zeros((0, 6)))

        if person_dets.size > 0:
            person_dets[:, [0, 2]] *= scale_x  # x coordinates
            person_dets[:, [1, 3]] *= scale_y  # y coordinates

        if phone_dets.size > 0:
            phone_dets[:, [0, 2]] *= scale_x  # x coordinates
            phone_dets[:, [1, 3]] *= scale_y  # y coordinates

        # Tracking
        if self.tracker:
            tracks = self.tracker.update(person_dets)
        else:
            tracks = {i+1: {"bbox": d[:4], "trace": [((d[0]+d[2])/2, (d[1]+d[3])/2)], "attrs": {}}
                      for i, d in enumerate(person_dets)}

        # Analytics (only if we have tracks)
        stats = {"inside_count": 0, "active_count": 0,
                 "loiter_count": 0, "speeds": {}}
        phone_present_count = 0
        phone_in_use_count = 0

        if tracks:
            # Phone usage analysis
            if self.cfg["branches"]["phone_usage"]:
                self.phone_logic.update(tracks, phone_dets)

            # People behavior analysis
            if self.cfg["branches"]["people_behavior"]:
                stats = self.analytics.update(tracks)

            # Calculate phone usage
            phone_present_count = sum(1 for _, t in tracks.items()
                                      if t.get("attrs", {}).get("phone_present", False))
            phone_in_use_count = sum(1 for _, t in tracks.items()
                                     if t.get("attrs", {}).get("phone_in_use", False))

        # Draw visualizations on original frame
        vis_frame = frame.copy()
        if self.cfg["output"]["draw"]["show_roi"]:
            draw_polygon(vis_frame, self.cfg["roi"]["polygon"])
        if self.cfg["output"]["draw"]["show_tracks"] and tracks:
            draw_tracks(vis_frame, tracks,
                        show_boxes=self.cfg["output"]["draw"]["show_boxes"],
                        show_labels=self.cfg["output"]["draw"]["show_labels"])
        if self.cfg["output"]["draw"]["show_counters"]:
            draw_counters(vis_frame, {
                "inside": stats["inside_count"],
                "active": stats["active_count"],
                "loiter": stats["loiter_count"],
                "phone_present": int(phone_present_count),
                "phone_in_use": int(phone_in_use_count),
            })

        # Add FPS counter
        process_time = time.time() - start_time
        fps = 1.0 / process_time if process_time > 0 else 0
        cv2.putText(vis_frame, f'Processing FPS: {fps:.1f}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Emit to PostgreSQL if configured
        if self.emitter:
            try:
                ts_ms = int(time.time() * 1000)
                self.emitter.emit_frame(ts_ms, tracks, {
                    "inside": stats["inside_count"],
                    "active": stats["active_count"],
                    "loiter": stats["loiter_count"],
                    "phone_present": int(phone_present_count),
                    "phone_in_use": int(phone_in_use_count),
                    "total_people": len(tracks)
                })
            except Exception as e:
                logging.error(f"Error emitting to PostgreSQL: {e}")

        # Update global state
        current_time = time.time()
        new_stats = {
            "timestamp": datetime.now().isoformat(),
            "inside_count": stats["inside_count"],
            "active_count": stats["active_count"],
            "loiter_count": stats["loiter_count"],
            "phone_present": int(phone_present_count),
            "phone_in_use": int(phone_in_use_count),
            "total_people": len(tracks),
            "processing_fps": round(fps, 1)
        }

        with frame_lock:
            current_frame = vis_frame
            current_stats = new_stats
            stats_history.append(new_stats.copy())

        return vis_frame, new_stats


# Global tracker instance
tracker = None


def tracking_worker():
    """Optimized background thread for processing video frames"""
    global tracker, system_running, last_emit_time

    while system_running:
        try:
            if tracker:
                frame, stats = tracker.process_frame()

                # Emit data with rate limiting
                current_time = time.time()
                if stats and (current_time - last_emit_time) >= EMIT_INTERVAL:
                    socketio.emit('live_data', stats)
                    last_emit_time = current_time

                # Small sleep to prevent CPU overload
                time.sleep(0.05)  # 20 FPS max processing
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
            tracker = OptimizedProductivityTracker()
            tracker.start()
            system_running = True

            # Start background processing thread
            tracker_thread = threading.Thread(
                target=tracking_worker, daemon=True)
            tracker_thread.start()

        return jsonify({"status": "success", "message": "Optimized tracking started"})
    except Exception as e:
        logging.error(f"Error starting tracking: {e}")
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
    """Optimized video streaming route with better compression"""
    def generate():
        global current_frame
        while True:
            with frame_lock:
                if current_frame is not None:
                    # Resize for streaming if too large
                    stream_frame = current_frame
                    h, w = stream_frame.shape[:2]
                    if w > 800:  # Limit streaming resolution
                        ratio = 800 / w
                        new_h = int(h * ratio)
                        stream_frame = cv2.resize(stream_frame, (800, new_h))

                    # Encode with better compression
                    encode_params = [
                        cv2.IMWRITE_JPEG_QUALITY, 75,  # Reduced quality for speed
                        cv2.IMWRITE_JPEG_OPTIMIZE, 1
                    ]
                    ret, buffer = cv2.imencode(
                        '.jpg', stream_frame, encode_params)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)  # 10 FPS streaming

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/current_stats')
def get_current_stats():
    """Get current real-time statistics"""
    global current_stats
    return jsonify(current_stats)


@app.route('/api/stats_history')
def get_stats_history():
    """Get historical statistics with limit"""
    global stats_history
    limit = request.args.get('limit', 50, type=int)  # Reduced default limit
    return jsonify(list(stats_history)[-limit:])


@app.route('/api/summary')
def get_summary():
    """Get optimized summary statistics"""
    try:
        if not stats_history:
            return jsonify({
                "current_session": {},
                "system_status": "stopped",
                "last_update": "",
                "total_data_points": 0
            })

        # Calculate summary from recent data
        recent_data = list(stats_history)[-100:]  # Only use last 100 points

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
            avg_fps = sum(d.get('processing_fps', 0)
                          for d in data) / total_points

            return {
                "avg_people_inside": round(avg_inside, 1),
                "avg_active_people": round(avg_active, 1),
                "avg_loitering": round(avg_loiter, 1),
                "avg_phone_usage": round(avg_phone_use, 1),
                "avg_processing_fps": round(avg_fps, 1),
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
    emit('status', {'message': 'Connected to Optimized Productivity Tracker'})


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)

    # Run with optimized settings
    socketio.run(app, debug=False, host='0.0.0.0', port=5000,
                 use_reloader=False, log_output=False)
