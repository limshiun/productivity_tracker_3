from typing import Dict, List, Tuple
import numpy as np
from shapely.geometry import Point, Polygon
class ROIAnalytics:
    def __init__(self, polygon_pts: List[List[float]], fps: float, loiter_seconds: float, active_speed_px_per_s: float):
        self.poly = Polygon(polygon_pts); self.fps=fps
        self.loiter_frames=int(max(1.0, loiter_seconds)*fps); self.active_speed=active_speed_px_per_s
        self.inside_frames: Dict[int,int]={}; self.last_center: Dict[int,Tuple[float,float]]={}; self.speed_px_per_s: Dict[int,float]={}
        self.active_ids=set(); self.loiter_ids=set()
    def update(self, tracks: Dict[int, dict]):
        self.active_ids.clear(); self.loiter_ids.clear()
        for tid,t in tracks.items():
            cx,cy=t["trace"][-1]; inside=self.poly.contains(Point(cx,cy))
            if inside: self.inside_frames[tid]=self.inside_frames.get(tid,0)+1
            else: self.inside_frames[tid]=0
            if tid in self.last_center:
                lx,ly=self.last_center[tid]; dist=np.hypot(cx-lx, cy-ly); speed=dist*self.fps
            else: speed=0.0
            self.speed_px_per_s[tid]=speed; self.last_center[tid]=(cx,cy)
            if self.inside_frames[tid]>=self.loiter_frames and speed < self.active_speed*0.5: self.loiter_ids.add(tid)
            elif speed>=self.active_speed and self.inside_frames[tid]>0: self.active_ids.add(tid)
        for tid in [tid for tid in list(self.inside_frames.keys()) if tid not in tracks]:
            self.inside_frames.pop(tid,None); self.last_center.pop(tid,None); self.speed_px_per_s.pop(tid,None)
        return {"active_count":len(self.active_ids),"loiter_count":len(self.loiter_ids),"inside_count":sum(1 for tid,t in tracks.items() if self.poly.contains(Point(t["trace"][-1]))),"speeds":dict(self.speed_px_per_s)}
