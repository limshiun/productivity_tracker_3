from typing import Dict
import numpy as np, itertools
def iou(a, b):
    xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1); h = max(0.0, yy2 - yy1)
    inter = w*h; areaA = (a[2]-a[0])*(a[3]-a[1]); areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-9)
class CentroidTracker:
    def __init__(self, fps: float, max_lost_secs: float = 2.0, min_iou_match: float = 0.2):
        self.next_id=1; self.tracks: Dict[int, dict]={}; self.fps=max(1.0,fps)
        self.max_lost_frames=int(max_lost_secs*self.fps); self.min_iou=min_iou_match
    def update(self, detections: np.ndarray):
        updated=set(); unmatched=list(range(len(detections)))
        if len(detections) and len(self.tracks):
            det_boxes=detections[:,:4]; tids=list(self.tracks.keys()); tboxes=np.array([self.tracks[i]['bbox'] for i in tids])
            ious=np.zeros((len(det_boxes),len(tboxes))); 
            for i in range(len(det_boxes)):
                for j in range(len(tboxes)): ious[i,j]=iou(det_boxes[i], tboxes[j])
            used=set()
            for i,j in sorted(itertools.product(range(len(det_boxes)), range(len(tboxes))), key=lambda x:-ious[x[0],x[1]]):
                if i not in unmatched: continue
                tid=tids[j]
                if tid in used: continue
                if ious[i,j] < self.min_iou: continue
                self._update_track(tid, detections[i]); updated.add(tid); used.add(tid); unmatched.remove(i)
        for i in unmatched:
            d=detections[i]; tid=self.next_id; self.next_id+=1
            self.tracks[tid]={"bbox":d[:4].astype(float),"cls":int(d[5]) if len(d)>5 else -1,"conf":float(d[4]) if len(d)>4 else 0.0,"trace":[self._center(d[:4])],"lost":0,"attrs":{}}
            updated.add(tid)
        todel=[]
        for tid,t in self.tracks.items():
            if tid not in updated: t["lost"]+=1
            if t["lost"]>self.max_lost_frames: todel.append(tid)
        for tid in todel: del self.tracks[tid]
        return self.tracks
    def _center(self, b):
        x1,y1,x2,y2=b; return ((x1+x2)/2.0, (y1+y2)/2.0)
    def _update_track(self, tid, det):
        t=self.tracks[tid]; t["bbox"]=det[:4].astype(float); t["conf"]=float(det[4]) if len(det)>4 else 0.0; t["cls"]=int(det[5]) if len(det)>5 else -1; t["trace"].append(self._center(det[:4])); t["lost"]=0
