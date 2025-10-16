"""
Headless runner for 24/7 operation with automatic scheduling.
"""
import logging
import time
import signal
import sys
import threading
import numpy as np
from typing import Optional
from .config import load_config
from .video_stream import VideoStream
from .detector import Detector
from .trackers.centroid import CentroidTracker
from .analytics import ROIAnalytics
from .behaviors import PhoneUsage, PPECompliance, FoodContainers, IdleObjects
from .emitter import JSONLEmitter
from .scheduler import DailyScheduler, create_scheduler_from_config


class HeadlessRunner:
    """
    Headless runner that operates 24/7 with automatic daily scheduling.
    """

    def __init__(self, config_path: str):
        """
        Initialize the headless runner.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.cfg = load_config(config_path)

        # Initialize components
        self.vs: Optional[VideoStream] = None
        self.detector: Optional[Detector] = None
        self.tracker: Optional[CentroidTracker] = None
        self.analytics: Optional[ROIAnalytics] = None
        self.phone_logic: Optional[PhoneUsage] = None
        self.ppe_logic: Optional[PPECompliance] = None
        self.food_logic: Optional[FoodContainers] = None
        self.idle_logic: Optional[IdleObjects] = None
        self.emitter: Optional[JSONLEmitter] = None
        self.scheduler: Optional[DailyScheduler] = None

        # State management
        self.is_capturing = False
        self.capture_thread: Optional[threading.Thread] = None
        self.stop_capture_event = threading.Event()

        # Setup logging
        self._setup_logging()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self):
        """Setup logging configuration."""
        level = self.cfg.get("logging", {}).get("level", "INFO")
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("productivity_tracker.log")
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)

    def _load_onnx_session(self, path):
        """Load ONNX session for PPE compliance."""
        if not path:
            return None
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            return ort.InferenceSession(path, sess_options=opts, providers=['CPUExecutionProvider'])
        except Exception as e:
            self.logger.warning(f"Failed to load ONNX model {path}: {e}")
            return None

    def _initialize_components(self):
        """Initialize all processing components."""
        self.logger.info("Initializing components...")

        # Video stream
        self.vs = VideoStream(
            source=self.cfg["input"]["source"],
            loop_file=self.cfg["input"]["loop_file"],
            read_timeout_sec=self.cfg["input"]["read_timeout_sec"],
            reconnect_delay_sec=self.cfg["input"]["reconnect_delay_sec"]
        )

        # Detector
        self.detector = Detector(self.cfg["models"])

        # Tracker (if enabled)
        if self.cfg["tracking"]["enabled"]:
            fps_cap = self.vs.fps() or self.cfg["analytics"]["fps_assumed"]
            self.tracker = CentroidTracker(
                fps=fps_cap,
                max_lost_secs=self.cfg["tracking"]["max_lost_secs"],
                min_iou_match=self.cfg["tracking"]["min_iou_match"]
            )

        # Analytics components
        fps_cap = self.vs.fps() or self.cfg["analytics"]["fps_assumed"]
        self.analytics = ROIAnalytics(
            polygon_pts=self.cfg["roi"]["polygon"],
            fps=fps_cap,
            loiter_seconds=self.cfg["analytics"]["loiter_seconds"],
            active_speed_px_per_s=self.cfg["analytics"]["active_speed_px_per_s"]
        )

        # Behavior detection
        self.phone_logic = PhoneUsage(
            overlap_iou=self.cfg["analytics"]["phone_overlap_iou"]
        )

        # PPE compliance
        mask_sess = self._load_onnx_session(
            self.cfg.get("ppe_classifiers", {}).get("mask_onnx")
        )
        sleeve_sess = self._load_onnx_session(
            self.cfg.get("ppe_classifiers", {}).get("sleeves_onnx")
        )
        self.ppe_logic = PPECompliance(
            mask_sess, sleeve_sess,
            run_every_n=self.cfg.get("ppe_classifiers", {}).get(
                "run_every_n_frames", 5),
            ema_alpha=self.cfg.get("ppe_classifiers", {}
                                   ).get("ema_alpha", 0.3),
            debug_draw=False  # Always false in headless mode
        )

        # Food containers
        self.food_logic = FoodContainers(
            min_speed_px_per_s=self.cfg["analytics"]["food_min_speed_px_per_s"]
        )

        # Idle objects (if enabled)
        if self.cfg["branches"].get("idle_objects", False):
            self.idle_logic = IdleObjects(
                fps=fps_cap,
                min_iou=self.cfg["analytics"].get("idle_min_iou", 0.3),
                max_lost_secs=self.cfg["analytics"].get(
                    "idle_max_lost_secs", 2.0),
                min_dwell_sec=self.cfg["analytics"].get(
                    "idle_min_dwell_sec", 2.0)
            )

        # Event emitter
        events_cfg = self.cfg["output"].get("events")
        if events_cfg:
            self.emitter = JSONLEmitter(
                config=events_cfg,
                aggregate_window_sec=self.cfg["output"].get(
                    "jsonl", {}).get("aggregate_window_sec", 60)
            )
        elif self.cfg["output"].get("jsonl", {}).get("enabled", False):
            self.emitter = JSONLEmitter(
                self.cfg["output"]["jsonl"]["path"],
                aggregate_window_sec=self.cfg["output"]["jsonl"]["aggregate_window_sec"]
            )

        self.logger.info("Components initialized successfully")

    def _capture_loop(self):
        """Main capture loop that runs during active hours."""
        self.logger.info("Starting capture loop...")

        # Open video stream
        self.vs.open()
        fps_cap = self.vs.fps() or self.cfg["analytics"]["fps_assumed"]

        self.logger.info(f"Stream opened at {fps_cap:.2f} FPS")

        prev_time = 0
        frame_count = 0

        while not self.stop_capture_event.is_set():
            try:
                ok, frame = self.vs.read()
                if not ok or frame is None:
                    self.logger.warning("Frame read failed, continuing...")
                    continue

                frame_count += 1

                # Single detector pass → per-branch slices
                dets = self.detector.infer(frame)
                person_dets = dets.get("person", np.zeros((0, 6)))
                phone_dets = dets.get("phone", np.zeros((0, 6)))
                food_dets = dets.get("food", np.zeros((0, 6)))
                object_dets = dets.get("objects", np.zeros((0, 6)))

                # Filter out person (0) and phone (67) from object_dets
                if object_dets.size > 0:
                    cls_col = object_dets[:, 5].astype(np.int32)
                    mask = (cls_col != 0) & (cls_col != 67)
                    object_dets = object_dets[mask]

                # People tracking
                if self.tracker:
                    tracks = self.tracker.update(person_dets)
                else:
                    tracks = {
                        i+1: {
                            "bbox": d[:4],
                            "trace": [((d[0]+d[2])/2, (d[1]+d[3])/2)],
                            "attrs": {}
                        }
                        for i, d in enumerate(person_dets)
                    }

                # Process branches
                if self.cfg["branches"]["phone_usage"]:
                    _ = self.phone_logic.update(tracks, phone_dets)

                # Derive counts from track attrs
                phone_present_count = sum(1 for _, t in tracks.items()
                                          if t.get("attrs", {}).get("phone_present", False))
                phone_in_use_count = sum(1 for _, t in tracks.items()
                                         if t.get("attrs", {}).get("phone_in_use", False))

                if self.cfg["branches"]["ppe_compliance"]:
                    self.ppe_logic.update(frame, tracks)

                if self.cfg["branches"]["people_behavior"]:
                    stats = self.analytics.update(tracks)
                else:
                    stats = {
                        "inside_count": 0,
                        "active_count": 0,
                        "loiter_count": 0,
                        "speeds": {}
                    }

                if self.cfg["branches"]["food_containers"]:
                    moved_containers = self.food_logic.update(
                        fps_cap, food_dets)
                else:
                    moved_containers = 0

                idle_stats = {"idle_objects": 0, "idle_max_dwell_sec": 0.0}
                if self.idle_logic is not None:
                    try:
                        idle_stats = self.idle_logic.update(
                            object_dets, self.cfg["roi"].get(
                                "empty_areas", []) or []
                        )
                    except Exception as e:
                        self.logger.debug(f"IdleObjects update failed: {e}")

                # Emit events
                if self.emitter:
                    ts_ms = int(time.time() * 1000)
                    self.emitter.emit_frame(ts_ms, tracks, {
                        "inside": stats["inside_count"],
                        "active": stats["active_count"],
                        "loiter": stats["loiter_count"],
                        "phone_present": int(phone_present_count),
                        "phone_in_use": int(phone_in_use_count),
                        "containers_moved": int(moved_containers),
                        "idle_objects": int(idle_stats.get("idle_objects", 0)),
                        "idle_max_dwell_sec": float(idle_stats.get("idle_max_dwell_sec", 0.0)),
                    })

                # Log progress every 1000 frames
                if frame_count % 1000 == 0:
                    self.logger.info(f"Processed {frame_count} frames")

                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time else 0
                prev_time = curr_time

                # Log FPS every 100 frames
                if frame_count % 100 == 0:
                    self.logger.debug(f"FPS: {fps:.1f}")

            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(1)  # Brief pause before retrying

        # Cleanup
        if self.vs:
            self.vs.close()

        self.logger.info("Capture loop ended")

    def start_capture(self):
        """Start the capture process."""
        if self.is_capturing:
            self.logger.warning("Capture is already running")
            return

        self.logger.info("Starting capture...")
        self.is_capturing = True
        self.stop_capture_event.clear()

        # Initialize components if not already done
        if not self.vs:
            self._initialize_components()

        # Start capture in a separate thread
        self.capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop_capture(self):
        """Stop the capture process."""
        if not self.is_capturing:
            return

        self.logger.info("Stopping capture...")
        self.is_capturing = False
        self.stop_capture_event.set()

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=10)

        self.logger.info("Capture stopped")

    def run(self):
        """Run the headless system with scheduling."""
        self.logger.info("Starting headless productivity tracker...")

        # Create and configure scheduler
        self.scheduler = create_scheduler_from_config(self.cfg)
        self.scheduler.set_capture_function(
            self.start_capture, self.stop_capture)

        # Start scheduler
        self.scheduler.start()

        # Log initial status
        status = self.scheduler.get_status()
        self.logger.info(f"Scheduler status: {status}")

        try:
            # Main loop - just keep the process alive
            while True:
                time.sleep(60)  # Check every minute

                # Log status every hour
                if int(time.time()) % 3600 < 60:
                    status = self.scheduler.get_status()
                    self.logger.info(f"Hourly status: {status}")

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        finally:
            self.shutdown()

    def shutdown(self):
        """Shutdown the system gracefully."""
        self.logger.info("Shutting down...")

        # Stop scheduler
        if self.scheduler:
            self.scheduler.stop()

        # Stop capture
        self.stop_capture()

        # Close video stream
        if self.vs:
            self.vs.close()

        self.logger.info("Shutdown complete")


def main():
    """Main entry point for headless operation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Headless Productivity Tracker")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()

    runner = HeadlessRunner(args.config)
    runner.run()


if __name__ == "__main__":
    main()
