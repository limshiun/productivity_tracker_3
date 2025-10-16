from typing import Dict, Any, Optional
import numpy as np
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
class Detector:
    def __init__(self, model_cfgs: Dict[str, Any]):
        self.models: Dict[str, Optional[object]] = {}
        for key, mc in model_cfgs.items():
            weights = mc.get("weights")
            if weights:
                if YOLO is None: raise RuntimeError("Ultralytics not available but weights specified.")
                self.models[key] = YOLO(weights)
            else:
                self.models[key] = None
        self.model_cfgs = model_cfgs
    def infer(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for key, model in self.models.items():
            cfg = self.model_cfgs[key]
            if model is None:
                out[key] = np.zeros((0,6), dtype=float); continue
            res = model.predict(image, conf=cfg.get("conf", 0.25), iou=cfg.get("iou", 0.5), verbose=False)
            boxes = res[0].boxes
            if boxes is None:
                out[key] = np.zeros((0,6), dtype=float); continue
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
            conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
            cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
            det = np.concatenate([xyxy, conf.reshape(-1,1), cls.reshape(-1,1)], axis=1)
            allowed = cfg.get("classes", [])
            if allowed: det = det[np.isin(det[:,5], allowed)]
            out[key] = det
        return out
