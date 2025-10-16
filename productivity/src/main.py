import argparse
import logging
import time
import cv2
import numpy as np
import time
from src.yolo_v12.config import load_config
from src.yolo_v12.video_stream import VideoStream
from src.yolo_v12.detector import Detector
from src.yolo_v12.trackers.centroid import CentroidTracker
from src.yolo_v12.analytics import ROIAnalytics
from src.yolo_v12.behaviors import PhoneUsage, PPECompliance, FoodContainers, IdleObjects
from src.yolo_v12.visualize import draw_polygon, draw_tracks, draw_counters
from src.yolo_v12.emitter import JSONLEmitter


def build_logger(level: str):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")


def load_onnx_session(path):
    if not path:
        return None
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        return ort.InferenceSession(path, sess_options=opts, providers=['CPUExecutionProvider'])
    except Exception as e:
        logging.warning(f"Failed to load ONNX model {path}: {e}")
        return None


def main(args):
    cfg = load_config(args.config)
    build_logger(cfg["logging"]["level"])

    # Apply sentry mode overrides
    sentry = cfg.get("output", {}).get("sentry_mode", {}) or {}
    if sentry.get("enabled", False):
        # Force headless
        cfg["output"]["show_window"] = False
        cfg["output"]["save_video"] = False
        # Route events to logging if requested
        if sentry.get("log_events", True) and cfg.get("output", {}).get("events"):
            for k in ("track_state", "summary"):
                if k in cfg["output"]["events"]:
                    cfg["output"]["events"][k]["mode"] = "log"

    # Input
    vs = VideoStream(source=cfg["input"]["source"],
                     loop_file=cfg["input"]["loop_file"],
                     read_timeout_sec=cfg["input"]["read_timeout_sec"],
                     reconnect_delay_sec=cfg["input"]["reconnect_delay_sec"])
    vs.open()
    fps_cap = vs.fps() or cfg["analytics"]["fps_assumed"]

    logging.info(
        f"Stream opened {cfg['input']['source']} at {fps_cap:.2f} FPS")

    # Detector (single pass → multi-branch fan-out)
    detector = Detector(cfg["models"])

    # Tracking (for persons)
    tracker = None
    if cfg["tracking"]["enabled"]:
        tracker = CentroidTracker(fps=fps_cap,
                                  max_lost_secs=cfg["tracking"]["max_lost_secs"],
                                  min_iou_match=cfg["tracking"]["min_iou_match"])

    # Branch logic
    analytics = ROIAnalytics(polygon_pts=cfg["roi"]["polygon"],
                             fps=fps_cap,
                             loiter_seconds=cfg["analytics"]["loiter_seconds"],
                             active_speed_px_per_s=cfg["analytics"]["active_speed_px_per_s"])

    phone_logic = PhoneUsage(overlap_iou=cfg["analytics"]["phone_overlap_iou"])

    mask_sess = load_onnx_session(
        cfg.get("ppe_classifiers", {}).get("mask_onnx"))
    sleeve_sess = load_onnx_session(
        cfg.get("ppe_classifiers", {}).get("sleeves_onnx"))
    ppe_logic = PPECompliance(mask_sess, sleeve_sess,
                              run_every_n=cfg.get("ppe_classifiers", {}).get(
                                  "run_every_n_frames", 5),
                              ema_alpha=cfg.get("ppe_classifiers", {}).get(
                                  "ema_alpha", 0.3),
                              debug_draw=cfg["output"]["draw"].get("show_boxes", False))

    food_logic = FoodContainers(
        min_speed_px_per_s=cfg["analytics"]["food_min_speed_px_per_s"])

    idle_logic = None
    if cfg["branches"].get("idle_objects", False):
        idle_logic = IdleObjects(
            fps=fps_cap,
            min_iou=cfg["analytics"].get("idle_min_iou", 0.3),
            max_lost_secs=cfg["analytics"].get("idle_max_lost_secs", 2.0),
            min_dwell_sec=cfg["analytics"].get("idle_min_dwell_sec", 2.0)
        )

    # Emitter JSONL events
    emitter = None
    # Prefer new granular events config if present; fallback to legacy jsonl
    events_cfg = cfg["output"].get("events")
    if events_cfg:
        emitter = JSONLEmitter(config=events_cfg,
                               aggregate_window_sec=cfg["output"].get("jsonl", {}).get("aggregate_window_sec", 60))
    elif cfg["output"].get("jsonl", {}).get("enabled", False):
        emitter = JSONLEmitter(cfg["output"]["jsonl"]["path"],
                               aggregate_window_sec=cfg["output"]["jsonl"]["aggregate_window_sec"])

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    if cfg["output"]["save_video"]:
        w, h = vs.size()
        writer = cv2.VideoWriter(
            cfg["output"]["save_path"], fourcc, fps_cap, (w, h))

    win_name = "YOLO V12 Multi-Branch"
    if cfg["output"]["show_window"]:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    prev_time = 0  # For FPS

    while True:
        ok, frame = vs.read()
        if not ok or frame is None:
            logging.warning("Frame read failed, continuing...")
            continue

        # Single detector pass → per-branch slices
        dets = detector.infer(frame)
        person_dets = dets.get("person", np.zeros((0, 6)))
        phone_dets = dets.get("phone", np.zeros((0, 6)))
        food_dets = dets.get("food",  np.zeros((0, 6)))
        object_dets = dets.get("objects", np.zeros((0, 6)))
        # filter out person (0) and phone (67) from object_dets
        if object_dets.size > 0:
            cls_col = object_dets[:, 5].astype(np.int32)
            mask = (cls_col != 0) & (cls_col != 67)
            object_dets = object_dets[mask]

        # People tracking
        if tracker:
            tracks = tracker.update(person_dets)
        else:
            tracks = {i+1: {"bbox": d[:4], "trace": [((d[0]+d[2])/2, (d[1]+d[3])/2)], "attrs": {
            }} for i, d in enumerate(person_dets)}

        # Branches
        if cfg["branches"]["phone_usage"]:
            _ = phone_logic.update(tracks, phone_dets)
        else:
            pass

        # Derive counts from track attrs set by PhoneUsage
        phone_present_count = sum(1 for _, t in tracks.items() if t.get(
            "attrs", {}).get("phone_present", False))
        phone_in_use_count = sum(1 for _, t in tracks.items() if t.get(
            "attrs", {}).get("phone_in_use", False))

        if cfg["branches"]["ppe_compliance"]:
            ppe_logic.update(frame, tracks)

        if cfg["branches"]["people_behavior"]:
            stats = analytics.update(tracks)
        else:
            stats = {"inside_count": 0, "active_count": 0,
                     "loiter_count": 0, "speeds": {}}

        if cfg["branches"]["food_containers"]:
            moved_containers = food_logic.update(fps_cap, food_dets)
        else:
            moved_containers = 0

        idle_stats = {"idle_objects": 0, "idle_max_dwell_sec": 0.0}
        if idle_logic is not None:
            try:
                idle_stats = idle_logic.update(
                    object_dets, cfg["roi"].get("empty_areas", []) or [])
            except Exception as e:
                logging.debug(f"IdleObjects update failed: {e}")

        # Draw
        if cfg["output"]["draw"]["show_roi"]:
            # draw main ROI polygon
            draw_polygon(frame, cfg["roi"]["polygon"])
            # draw empty areas if any
            for area in cfg["roi"].get("empty_areas", []) or []:
                try:
                    poly = area.get("polygon") or area.get("points")
                    if isinstance(poly, (list, tuple)) and len(poly) >= 3:
                        # validate points
                        pts = []
                        for pt in poly:
                            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                                pts.append([float(pt[0]), float(pt[1])])
                        if len(pts) >= 3:
                            draw_polygon(frame, pts)
                except Exception:
                    # skip bad polygons
                    pass
        if cfg["output"]["draw"]["show_tracks"]:
            draw_tracks(frame, tracks,
                        show_boxes=cfg["output"]["draw"]["show_boxes"],
                        show_labels=cfg["output"]["draw"]["show_labels"])
        if cfg["output"]["draw"]["show_counters"]:
            draw_counters(frame, {
                "inside": stats["inside_count"],
                "active": stats["active_count"],
                "loiter": stats["loiter_count"],
                "phone_present": int(phone_present_count),
                "phone_in_use": int(phone_in_use_count),
                "containers_moved": int(moved_containers),
                "idle_objects": int(idle_stats.get("idle_objects", 0)),
                "idle_max_dwell_s": float(idle_stats.get("idle_max_dwell_sec", 0.0)),
            })

        # Emit JSONL
        if emitter:
            ts_ms = int(time.time() * 1000)
            emitter.emit_frame(ts_ms, tracks, {
                "inside": stats["inside_count"],
                "active": stats["active_count"],
                "loiter": stats["loiter_count"],
                "phone_present": int(phone_present_count),
                "phone_in_use": int(phone_in_use_count),
                "containers_moved": int(moved_containers),
                "idle_objects": int(idle_stats.get("idle_objects", 0)),
                "idle_max_dwell_sec": float(idle_stats.get("idle_max_dwell_sec", 0.0)),
            })

        # Calculate time difference and FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        # Overlay FPS counter
        fps_text = f'FPS: {fps:.1f}'
        cv2.putText(frame, fps_text, (10,
                    frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Video I/O and UI events
        if writer:
            writer.write(frame)
        if cfg["output"]["show_window"]:
            cv2.imshow(win_name, frame)
            # If window was closed by the user, exit
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            # slightly longer delay improves reliability
            key = cv2.waitKey(10) & 0xFF
            if key in (27, ord('q'), ord('Q')):  # Esc, q, or Q
                break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    args = p.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        logging.info("Interrupted by user, shutting down...")
