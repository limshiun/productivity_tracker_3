import cv2, time, os
class VideoStream:
    def __init__(self, source: str, loop_file: bool = False, read_timeout_sec: int = 8, reconnect_delay_sec: int = 3):
        self.source = source; self.loop_file = loop_file
        self.read_timeout_sec = read_timeout_sec; self.reconnect_delay_sec = reconnect_delay_sec
        self.cap=None; self._is_file = os.path.exists(source)
    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened(): raise RuntimeError(f"Failed to open source: {self.source}")
    def read(self):
        if self.cap is None: self.open()
        ok, frame = self.cap.read()
        if ok: return True, frame
        if self._is_file and self.loop_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read(); return ok, frame
        start = time.time(); self.cap.release(); time.sleep(self.reconnect_delay_sec); self.open()
        while time.time() - start < self.read_timeout_sec:
            ok, frame = self.cap.read()
            if ok: return True, frame
        return False, None
    def fps(self):
        if self.cap is None: return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS) or 0.0
    def size(self):
        if self.cap is None: return (0,0)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w,h)
    def release(self):
        if self.cap is not None: self.cap.release(); self.cap=None
