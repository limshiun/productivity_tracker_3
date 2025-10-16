from typing import Dict, Tuple, Optional
import numpy as np
import cv2

def iou_xyxy(a, b):
    xx1=max(a[0], b[0]); yy1=max(a[1], b[1])
    xx2=min(a[2], b[2]); yy2=min(a[3], b[3])
    w=max(0.0, xx2-xx1); h=max(0.0, yy2-yy1)
    inter=w*h; areaA=(a[2]-a[0])*(a[3]-a[1]); areaB=(b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-9)

class PhoneUsage:
    """Associates 'cell phone' detections to person tracks more robustly and flags phone_in_use.
       Uses IoU, IoA (intersection over phone area), normalized center distance, bbox dilation, and temporal smoothing."""

    def __init__(
            self,
            overlap_iou: float = 0.15,
            phone_ioa: float = 0.4,  # fraction of phone box covered by the (dilated) person box
            expand_person: float = 0.2,  # dilate person bbox by this ratio on each side
            max_center_norm: float = 0.7,  # normalized center distance threshold (relative to person diagonal)
            on_frames: int = 1,  # kept for backward-compat; overridden by near_seconds if provided
            off_frames: int = 3,  # frames to clear phone_in_use
            marker_on_frames: int = 30,  # sustained frames to show a long-duration usage marker
            marker_off_frames: int = 10,  # frames to clear the long-duration usage marker
            near_seconds: float = 3.0,  # if >0, number of seconds to deem as using phone
            fps: float = 30.0  # assumed FPS to translate seconds to frames
    ):
        self.th_iou = overlap_iou
        self.th_ioa = phone_ioa 
        self.expand = expand_person
        self.max_center_norm = max_center_norm
        # time-based threshold (3 seconds by default)
        self.fps = max(1e-3, float(fps))
        self.near_seconds = max(0.0, float(near_seconds))
        # If near_seconds>0, use that to set confirmation frames; otherwise fall back to on_frames
        if self.near_seconds > 0:
            self.on_frames = max(1, int(round(self.near_seconds * self.fps)))
        else:
            self.on_frames = max(1, on_frames)
        self.off_frames = max(1, off_frames)
        self.marker_on_frames = max(1, marker_on_frames)
        self.marker_off_frames = max(1, marker_off_frames)
        # Gentle decay for confirmation counter on misses (prevents hard reset to 0 on a single miss)
        self._on_miss_decay = 1
        # per-track debounce counters
        self._on_cnt = {}
        self._off_cnt = {}
        # marker decay counters (we reuse _on_cnt for the "on" accumulation)
        self._marker_off_cnt = {}
        # track phone presence episode state and duration accumulation (frames)
        self._present_state = {}
        self._present_frames = {}

    def _ioa_phone(self, phone, box):
        # Intersection over phone area
        xx1=max(phone[0], box[0]); yy1=max(phone[1], box[1])
        xx2=min(phone[2], box[2]); yy2=min(phone[3], box[3])
        w=max(0.0, xx2-xx1); h=max(0.0, yy2-yy1)
        inter = w*h
        areaP = max(1e-9, (phone[2]-phone[0])*(phone[3]-phone[1]))
        return inter / areaP

    def _dilate(self, box, img_w=None, img_h=None, r=0.2):
        x1,y1,x2,y2 = box
        w = x2 - x1
        h = y2 - y1
        dx = r * w
        dy = r * h
        nx1 = x1 - dx; ny1 = y1 - dy
        nx2 = x2 + dx; ny2 = y2 + dy
        if img_w is not None and img_h is not None:
            nx1 = max(0, nx1); ny1 = max(0, ny1)
            nx2 = min(img_w - 1, nx2); ny2 = min(img_h - 1, ny2)
        return (nx1, ny1, nx2, ny2)

    def _center(self, b):
        return ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)

    def _center_distance_norm(self, phone, person):
        pc = self._center(phone)
        hc = self._center(person)
        dx = pc[0]-hc[0]; dy = pc[1]-hc[1]
        diag = ((person[2]-person[0])**2 + (person[3]-person[1])**2) ** 0.5
        if diag <= 1e-6: return 1e9
        return ((dx*dx + dy*dy) ** 0.5) / diag

    def update(self, tracks: Dict[int, dict], phone_dets: np.ndarray):
        # Initialize attrs and debounce/presence structures
        for tid, t in tracks.items():
            t.setdefault("attrs", {})
            t["attrs"].setdefault("phone_in_use", False)
            t["attrs"].setdefault("phone_usage_marker", False)  # long-duration marker
            t["attrs"].setdefault("phone_present", False)       # immediate presence near the person
            t["attrs"].setdefault("phone_appear_count", 0)      # counts when phone first appears near the person
            t["attrs"].setdefault("phone_present_seconds", 0.0) # duration of the current presence episode
            self._on_cnt.setdefault(tid, 0)
            self._off_cnt.setdefault(tid, 0)
            self._marker_off_cnt.setdefault(tid, 0)
            self._present_state.setdefault(tid, False)
            self._present_frames.setdefault(tid, 0)

        if phone_dets is None or len(phone_dets) == 0 or len(tracks) == 0:
            # No phones: decay states
            for tid, t in tracks.items():
                # clear immediate presence episode if it was active
                if self._present_state.get(tid, False):
                    self._present_state[tid] = False
                    self._present_frames[tid] = 0
                    t["attrs"]["phone_present"] = False
                    t["attrs"]["phone_present_seconds"] = 0.0

                if t["attrs"].get("phone_in_use", False):
                    self._off_cnt[tid] += 1
                    if self._off_cnt[tid] >= self.off_frames:
                        t["attrs"]["phone_in_use"] = False
                        self._on_cnt[tid] = 0
                else:
                    # gentle decay instead of hard reset to 0 to allow brief gaps to not kill buildup
                    self._on_cnt[tid] = max(0, self._on_cnt[tid] - self._on_miss_decay)
            return sum(1 for _, t in tracks.items() if t["attrs"].get("phone_in_use", False))

        # Build candidate matches with a composite score, one best per phone
        # Score prioritizes IoA and IoU, then center proximity
        assignments = []  # list of tuples (phone_idx, best_tid, score, pass_gate)
        for pi, det in enumerate(phone_dets):
            pb = det[:4]
            best = (None, -1.0, False)
            for tid, t in tracks.items():
                person = t["bbox"]
                dil = self._dilate(person, None, None, self.expand)

                iou = iou_xyxy(pb, person)
                ioa = self._ioa_phone(pb, dil)
                cdist = self._center_distance_norm(pb, person)

                pass_gate = (iou >= self.th_iou) or (ioa >= self.th_ioa) or (cdist <= self.max_center_norm)

                # Composite score: weight IoA highest (since phone is small), then IoU, then inverse distance
                score = 0.6*ioa + 0.3*iou + 0.1*max(0.0, 1.0 - min(1.0, cdist))
                if score > best[1]:
                    best = (tid, score, pass_gate)
            assignments.append((pi, best[0], best[1], best[2]))

        # Optionally ensure one phone per person by picking the best per person
        # Create per-person winner by highest score among phones that pass gate
        per_person_best = {}
        for pi, tid, score, ok in assignments:
            if not ok or tid is None:
                continue
            cur = per_person_best.get(tid)
            if cur is None or score > cur[1]:
                per_person_best[tid] = (pi, score)

        # Update debounce counters and presence logic
        matched_tids = set(per_person_best.keys())
        for tid, t in tracks.items():
            if tid in matched_tids:
                # presence state and duration
                if not self._present_state.get(tid, False):
                    # first frame of a new presence episode
                    t["attrs"]["phone_appear_count"] = t["attrs"].get("phone_appear_count", 0) + 1
                self._present_state[tid] = True
                self._present_frames[tid] = self._present_frames.get(tid, 0) + 1
                t["attrs"]["phone_present"] = True
                t["attrs"]["phone_present_seconds"] = float(self._present_frames[tid]) / float(self.fps)

                # debounce for usage
                self._on_cnt[tid] += 1
                self._off_cnt[tid] = 0
                # short debounce flag (threshold derived from seconds)
                if self._on_cnt[tid] >= self.on_frames:
                    t["attrs"]["phone_in_use"] = True
                # long-duration marker: turns on after sustained on_cnt
                if self._on_cnt[tid] >= self.marker_on_frames:
                    t["attrs"]["phone_usage_marker"] = True
                # matched frame resets marker decay
                self._marker_off_cnt[tid] = 0
            else:
                # end of presence episode (if any)
                if self._present_state.get(tid, False):
                    self._present_state[tid] = False
                    self._present_frames[tid] = 0
                    t["attrs"]["phone_present"] = False
                    t["attrs"]["phone_present_seconds"] = 0.0

                if t["attrs"].get("phone_in_use", False):
                    self._off_cnt[tid] += 1
                    if self._off_cnt[tid] >= self.off_frames:
                        t["attrs"]["phone_in_use"] = False
                        self._on_cnt[tid] = 0
                else:
                    # gentle decay instead of hard reset to 0
                    self._on_cnt[tid] = max(0, self._on_cnt[tid] - self._on_miss_decay)
                # handle marker decay separately (require longer misses to clear)
                if t["attrs"].get("phone_usage_marker", False):
                    self._marker_off_cnt[tid] += 1
                    if self._marker_off_cnt[tid] >= self.marker_off_frames:
                        t["attrs"]["phone_usage_marker"] = False
                        # do not reset _on_cnt here; it's governed by short debounce


class EMA:
    def __init__(self, alpha: float): self.alpha=alpha
    def update(self, prev: Optional[float], value: float)->float:
        return value if prev is None else (self.alpha*value + (1.0-self.alpha)*prev)

class PPECompliance:
    """Runs lightweight classifiers on person crops. Accepts ONNX models producing a single logit or prob.
       If models are None, acts as a no-op (keeps attrs if present)."""
    def __init__(self, mask_sess, sleeve_sess, run_every_n: int = 5, ema_alpha: float = 0.3, debug_draw: bool = False):
        self.mask_sess = mask_sess; self.sleeve_sess = sleeve_sess
        self.run_every_n = max(1, run_every_n)
        self.frame_idx = 0
        self.ema_mask = {}; self.ema_sleeve = {}
        self.ema_m = EMA(ema_alpha); self.ema_s = EMA(ema_alpha)
        # For 2-class outputs, your model uses index 0 for "mask" (order: [mask, no_mask]).
        self.mask_index = 0
        # For single-output models, interpret the value directly as "mask" probability (no inversion).
        self.mask_invert_single = False
        # Debug: draw the head crop rectangle on the frame if enabled
        self.debug_draw = debug_draw


    def _infer_input_spec(self, sess):
        """Infer layout (NHWC/NCHW) and target H,W from an ONNX session input."""
        # Defaults if shape is dynamic/unknown
        layout = "NHWC"
        H = 224
        W = 224
        try:
            inp = sess.get_inputs()[0]
            shape = list(inp.shape)  # e.g., [None, 224, 224, 3] or [1, 3, 224, 224]
            if len(shape) >= 4:
                # Try to infer by channel position
                # Numbers may be 'None' for dynamic; guard with isinstance
                dim1 = shape[1] if isinstance(shape[1], int) else None
                dim2 = shape[2] if isinstance(shape[2], int) else None
                dim3 = shape[3] if isinstance(shape[3], int) else None

                if dim1 == 3:
                    layout = "NCHW"
                    H = dim2 or H
                    W = dim3 or W
                elif dim3 == 3:
                    layout = "NHWC"
                    H = dim1 or H
                    W = dim2 or W
                else:
                    # Fallback: assume NHWC if can't detect channel pos
                    if isinstance(shape[1], int): H = shape[1]
                    if isinstance(shape[2], int): W = shape[2]
        except Exception:
            pass
        return layout, int(H), int(W)

    def _prep_for_sess(self, sess, img):
        """Prepare input tensor for the given session (resize, layout, normalization)."""
        layout, H, W = self._infer_input_spec(sess)
        im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
        im = im.astype(np.float32) / 255.0
        if layout == "NCHW":
            im = im.transpose(2, 0, 1)  # CHW
        # Add batch dimension
        return np.expand_dims(im, 0)

    def _infer_binary(self, sess, inp)->float:
        # Returns a probability in [0,1] that "mask" is present, handling logits vs probs and 1- vs 2-class outputs
        if sess is None: return 0.0
        inp_name = sess.get_inputs()[0].name
        out = sess.run(None, {inp_name: inp})[0]

        flat = out.reshape(-1).astype(np.float32)
        if flat.size >= 2:
            # Assume multi-class; treat as [mask, no_mask] for 2-class models.
            # Apply softmax if not already a probability simplex
            v = flat[:max(2, self.mask_index+1)].copy()
            if (v < 0).any() or (v > 1).any() or not np.isclose(v.sum(), 1.0, atol=1e-3):
                v = np.exp(v - np.max(v))
                s = np.clip(v.sum(), 1e-9, None)
                v = v / s
            idx = 1 if self.mask_index not in (0, 1) else self.mask_index
            p = float(v[idx])
            return max(0.0, min(1.0, p))

        # Single output case: interpret as prob or logit for "mask"
        val = float(flat[0])
        # Convert logit to prob if it looks like a logit
        if val < 0 or val > 1:
            val = 1.0 / (1.0 + np.exp(-val))
        return max(0.0, min(1.0, val))
    def update(self, frame, tracks: Dict[int, dict]):
        self.frame_idx += 1
        do_run = (self.frame_idx % self.run_every_n == 0)
        if not do_run or (self.mask_sess is None and self.sleeve_sess is None):
            # ensure attrs exist
            for tid,t in tracks.items():
                t.setdefault("attrs", {})
                t["attrs"].setdefault("mask_prob", None)
                t["attrs"].setdefault("sleeve_prob", None)
            return

        h, w = frame.shape[:2]
        for tid, t in tracks.items():
            x1,y1,x2,y2 = map(int, t["bbox"])
            x1=max(0,x1); y1=max(0,y1); x2=min(w-1,x2); y2=min(h-1,y2)
            if x2<=x1 or y2<=y1: continue

            # Use a head-focused crop (top portion of the person box) for PPE classifiers
            bw = x2 - x1
            bh = y2 - y1
            head_y2 = y1 + int(0.45 * bh)  # top ~45% height
            head_x1 = x1 + int(0.2 * bw)   # tighten sides to reduce background
            head_x2 = x2 - int(0.2 * bw)
            head_x1 = max(0, min(head_x2-1, head_x1))
            head_x2 = max(head_x1+1, min(w-1, head_x2))
            head_y2 = max(y1+1, min(h-1, head_y2))

            # Draw the head crop boundary if debug is enabled
            if self.debug_draw:
                cv2.rectangle(frame, (head_x1, y1), (head_x2, head_y2), (0, 255, 255), 2)

            crop = frame[y1:head_y2, head_x1:head_x2]

            t.setdefault("attrs", {})
            # mask
            if self.mask_sess is not None:
                inp_mask = self._prep_for_sess(self.mask_sess, crop)
                p = self._infer_binary(self.mask_sess, inp_mask)
                prev = self.ema_mask.get(tid)
                p = self.ema_m.update(prev, p)
                self.ema_mask[tid] = p
                t["attrs"]["mask_prob"] = p
            # sleeves
            if self.sleeve_sess is not None:
                inp_sleeve = self._prep_for_sess(self.sleeve_sess, crop)
                p = self._infer_binary(self.sleeve_sess, inp_sleeve)
                prev = self.ema_sleeve.get(tid)
                p = self.ema_s.update(prev, p)
                self.ema_sleeve[tid] = p
                t["attrs"]["sleeve_prob"] = p



class FoodContainers:
    """Tracks container detections and estimates moving count in ROI by speed threshold."""
    def __init__(self, min_speed_px_per_s: float = 30.0):
        self.min_speed = min_speed_px_per_s
        self.last_centers = {}
        self.speeds = {}

    def update(self, fps: float, dets: np.ndarray):
        moved = 0
        for i, d in enumerate(dets):
            x1,y1,x2,y2 = d[:4]
            cx,cy = (x1+x2)/2.0, (y1+y2)/2.0
            prev = self.last_centers.get(i)
            if prev is not None:
                dist = ((cx-prev[0])**2 + (cy-prev[1])**2) ** 0.5
                speed = dist * fps
                self.speeds[i] = speed
                if speed >= self.min_speed: moved += 1
            self.last_centers[i] = (cx,cy)
        return moved

class IdleObjects:
    """Detects and tracks non-human objects lingering inside configured empty areas (polygons).
    Provides per-frame summary: total objects inside and max dwell seconds among them.
    Matching is simple IoU-based with short persistence gaps.
    """
    def __init__(self, fps: float, min_iou: float = 0.3, max_lost_secs: float = 2.0, min_dwell_sec: float = 2.0):
        self.fps = max(1e-3, float(fps))
        self.min_iou = float(min_iou)
        self.max_lost_frames = max(1, int(self.fps * float(max_lost_secs)))
        self.min_dwell_frames = max(1, int(self.fps * float(min_dwell_sec)))
        # state per ROI name
        self._tracks = {}  # roi_name -> {id: {bbox, cls, dwell_frames, lost_frames}}
        self._next_id = {} # roi_name -> next int id

    def _ensure_roi(self, roi_name: str):
        if roi_name not in self._tracks:
            self._tracks[roi_name] = {}
            self._next_id[roi_name] = 1

    @staticmethod
    def _bbox_center(b):
        return ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)

    @staticmethod
    def _iou(a, b):
        return iou_xyxy(a, b)

    @staticmethod
    def _inside_polygon(bbox, polygon_np: np.ndarray) -> bool:
        # test bbox center inside polygon
        cx, cy = ((bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0)
        return cv2.pointPolygonTest(polygon_np, (float(cx), float(cy)), False) >= 0

    def update(self, dets: np.ndarray, empty_areas: list) -> Dict[str, float]:
        """Update tracking given detections and list of empty areas.
        empty_areas: list of dicts with keys {name, polygon}
        dets: np.ndarray shape Nx6 [x1,y1,x2,y2,conf,cls]
        Returns per-frame summary across all empty areas.
        """
        total_inside = 0
        max_dwell_frames = 0
        # Prepare polygons
        areas = []
        for area in empty_areas or []:
            try:
                if not isinstance(area, dict):
                    continue
                name = str(area.get("name") or "empty")
                raw_poly = area.get("polygon") or area.get("points") or []
                # Expect list of [x, y]
                if not isinstance(raw_poly, (list, tuple)):
                    continue
                pts = []
                for pt in raw_poly:
                    if isinstance(pt, (list, tuple)) and len(pt) == 2:
                        try:
                            x = float(pt[0]); y = float(pt[1])
                            pts.append([x, y])
                        except Exception:
                            pass
                if len(pts) < 3:
                    continue
                poly = np.array(pts, dtype=np.float32)
                areas.append((name, poly))
            except Exception:
                # skip malformed entries
                continue
        # For each area, match and update
        for name, poly in areas:
            self._ensure_roi(name)
            # filter dets whose center is inside this poly
            inside_idx = []
            inside_boxes = []
            inside_cls = []
            for i, d in enumerate(dets):
                b = d[:4]
                if self._inside_polygon(b, poly):
                    inside_idx.append(i); inside_boxes.append(b); inside_cls.append(int(d[5]))
            # match to existing tracks by IoU
            tracks = self._tracks[name]
            used = set()
            # first, attempt to match
            for tid, tr in list(tracks.items()):
                best_j = -1; best_iou = 0.0
                for j, b in enumerate(inside_boxes):
                    if j in used: continue
                    iou = self._iou(tr["bbox"], b)
                    if iou > best_iou:
                        best_iou = iou; best_j = j
                if best_j >= 0 and best_iou >= self.min_iou:
                    # matched
                    tr["bbox"] = inside_boxes[best_j]
                    tr["cls"] = inside_cls[best_j]
                    tr["dwell_frames"] += 1
                    tr["lost_frames"] = 0
                    used.add(best_j)
                else:
                    tr["lost_frames"] += 1
                    if tr["lost_frames"] > self.max_lost_frames:
                        tracks.pop(tid, None)
            # create new tracks for unmatched detections
            for j, b in enumerate(inside_boxes):
                if j in used: continue
                tid = self._next_id[name]
                self._next_id[name] += 1
                tracks[tid] = {"bbox": b, "cls": inside_cls[j], "dwell_frames": 1, "lost_frames": 0}
            # accumulate counts for this area
            for tid, tr in tracks.items():
                if tr["lost_frames"] == 0 and tr["dwell_frames"] >= 1:
                    total_inside += 1
                    if tr["dwell_frames"] > max_dwell_frames:
                        max_dwell_frames = tr["dwell_frames"]
        # Convert to seconds
        return {
            "idle_objects": int(total_inside),
            "idle_max_dwell_sec": float(max_dwell_frames) / float(self.fps)
        }
