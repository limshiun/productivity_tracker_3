from typing import Dict, List
import cv2, numpy as np
def draw_polygon(img, polygon: List[List[float]]):
    pts=np.array(polygon, dtype=np.int32); cv2.polylines(img,[pts],isClosed=True,color=(0,255,255),thickness=2); return img
def draw_tracks(img, tracks: Dict[int, dict], show_boxes=True, show_labels=True):
    for tid,t in tracks.items():
        x1,y1,x2,y2=map(int,t["bbox"])
        if show_boxes:
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        if show_labels:
            phone = ' 📱' if t.get('attrs',{}).get('phone_in_use', False) else ''
            maskp = t.get('attrs',{}).get('mask_prob', None)
            sleevep = t.get('attrs',{}).get('sleeve_prob', None)
            extras = []
            if maskp is not None: extras.append(f"M:{maskp:.2f}")
            if sleevep is not None: extras.append(f"S:{sleevep:.2f}")
            extra_txt = (" " + " ".join(extras)) if extras else ""
            label=f"ID {tid}{phone}{extra_txt}"
            cv2.putText(img,label,(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    return img
def draw_counters(img, counters: Dict[str, int]):
    y=24
    for k,v in counters.items():
        cv2.putText(img,f"{k}: {v}",(8,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2); y+=24
    return img
