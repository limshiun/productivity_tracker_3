# emitter.py
# Requires: pip install psycopg2-binary
import json
import time
import collections
import urllib.request
import logging
import os
from typing import Dict, Any, List, Optional

# Optional import (only needed if mode=pg is used)
try:
    import psycopg2
    from psycopg2 import sql
except Exception:
    psycopg2 = None
    sql = None


class _BaseSink:
    def send(self, obj: dict):
        raise NotImplementedError

    def close(self):
        pass


class _FileSink(_BaseSink):
    def __init__(self, path: str):
        # ensure directory exists
        dirpath = os.path.dirname(path) or "."
        os.makedirs(dirpath, exist_ok=True)
        self.path = path
        self.f = open(self.path, "a", encoding="utf-8")

    def send(self, obj: dict):
        self.f.write(json.dumps(obj) + "\n")
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except:
            pass


class _APISink(_BaseSink):
    def __init__(self, url: str, timeout: float = 2.5):
        self.url = url
        self.timeout = timeout

    def send(self, obj: dict):
        data = json.dumps(obj).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers={
            "Content-Type": "application/json"
        })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as _:
                pass
        except Exception:
            # Best-effort, avoid crashing pipeline on network errors
            pass


class _LogSink(_BaseSink):
    def __init__(self, level: int = logging.INFO):
        self.level = level

    def send(self, obj: dict):
        try:
            logging.log(self.level, json.dumps(obj))
        except Exception:
            pass


class _PGSink(_BaseSink):
    """
    PostgreSQL sink: writes one JSON document per row into table (JSONB column 'doc').
    Auto-creates the table if missing:
      id SERIAL PRIMARY KEY, ts TIMESTAMPTZ DEFAULT now(), doc JSONB NOT NULL
    DSN example: postgresql://postgres:1010@localhost:5432/events
    """

    def __init__(self, dsn: str, table: str = "raw_events"):
        if psycopg2 is None:
            raise RuntimeError(
                "psycopg2-binary is required for Postgres mode. Install with: pip install psycopg2-binary")
        self.table = table
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        self._ensure_table()

    def _ensure_table(self):
        create_stmt = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {tbl} (
                id SERIAL PRIMARY KEY,
                ts TIMESTAMPTZ DEFAULT now(),
                doc JSONB NOT NULL
            )
        """).format(tbl=sql.Identifier(self.table))
        self.cur.execute(create_stmt)

    def send(self, obj: dict):
        insert_stmt = sql.SQL("INSERT INTO {tbl} (doc) VALUES (%s)").format(
            tbl=sql.Identifier(self.table)
        )
        self.cur.execute(insert_stmt, [json.dumps(obj)])

    def close(self):
        try:
            self.cur.close()
            self.conn.close()
        except Exception:
            pass


class JSONLEmitter:
    """
    Emits events with separate control for track_state and summary.
    - Granular min interval seconds before sending per type.
    - Aggregates per-frame counts over the summary interval.
    - Supports sinks: file | api | log | pg
    Backward compatible with previous (single jsonl file, per-frame events).

    Constructor is compatible with your main.py:
      JSONLEmitter(config=..., client_id=..., client_name=..., location=...)

    Also supports legacy usage with just path=... for a single JSONL file.
    """

    def __init__(self,
                 path: Optional[str] = None,
                 aggregate_window_sec: int = 60,
                 config: Optional[dict] = None,
                 client_id: Optional[str] = None,
                 client_name: Optional[str] = None,
                 location: Optional[str] = None):

        # Rolling window for aggregates() API
        self.window = aggregate_window_sec
        self.events = collections.deque()  # (ts, summary_dict)

        # Metadata to attach to every event
        self._meta_out: Dict[str, Any] = {}

        # Default legacy behavior
        self._legacy = config is None
        if self._legacy:
            # Legacy: single file sink for both types
            self.track_sink: _BaseSink = _FileSink(path or "runs/events.jsonl")
            self.summary_sink: _BaseSink = self.track_sink
            self.track_enabled = True
            self.summary_enabled = True
            self.track_min_interval = 0.0
            self.summary_min_interval = 0.0
        else:
            ev = config or {}
            tr = ev.get("track_state", {}) or {}
            sm = ev.get("summary", {}) or {}
            # Enable flags
            self.track_enabled = bool(tr.get("enabled", True))
            self.summary_enabled = bool(sm.get("enabled", True))
            # Intervals
            self.track_min_interval = float(tr.get("min_interval_sec", 0))
            self.summary_min_interval = float(sm.get("min_interval_sec", 0))
            # Sinks
            self.track_sink = self._make_sink(tr)
            self.summary_sink = self._make_sink(sm)
            # Optional override for rolling window
            self.window = int(
                sm.get("aggregate_window_sec", aggregate_window_sec))
            # Meta from config
            meta_cfg = (ev.get("meta") or {}) if isinstance(
                ev.get("meta"), dict) else {}
            name = meta_cfg.get("name")
            cid = meta_cfg.get("id")
            loc = meta_cfg.get("location")
            if name is not None:
                self._meta_out["client_name"] = name
            if cid is not None:
                self._meta_out["client_id"] = cid
            if loc is not None:
                self._meta_out["location"] = loc

        # Meta overrides from constructor args (match your main.py call pattern)
        if client_name:
            self._meta_out["client_name"] = client_name
        if client_id:
            self._meta_out["client_id"] = client_id
        if location:
            self._meta_out["location"] = location

        # Timers and accumulators
        now = time.time()
        self._last_track_emit = 0.0
        self._last_summary_emit = 0.0
        self._summary_accum: Dict[str, int] = {}
        self._summary_frames = 0
        self._summary_first_ts_ms: Optional[int] = None

    def _make_sink(self, section: dict) -> _BaseSink:
        mode = (section.get("mode") or section.get("to") or "file").lower()

        if mode == "api":
            url = section.get("api_url") or section.get("url")
            if not url:
                # Fallback to file if URL missing
                mode = "file"
            else:
                return _APISink(url, timeout=float(section.get("timeout", 2.5)))

        if mode == "log":
            return _LogSink()

        if mode == "pg":
            dsn = section.get("dsn")
            table = section.get("table", "raw_events")
            if not dsn:
                raise ValueError(
                    "Postgres mode requires 'dsn' (e.g. postgresql://postgres:1010@localhost:5432/events)")
            return _PGSink(dsn=dsn, table=table)

        # file mode (default)
        path = section.get("file_path") or section.get(
            "path") or "runs/events.jsonl"
        return _FileSink(path)

    def emit_frame(self, ts_ms: int, tracks: Dict[int, dict], per_frame_counts: Dict[str, int], roi_name: str = "roi"):
        now = time.time()
        # Track state: throttle by interval
        if self.track_enabled and (now - self._last_track_emit >= self.track_min_interval):
            for tid, t in tracks.items():
                rec = {
                    "ts_ms": ts_ms,
                    "type": "track_state",
                    "id": tid,
                    "bbox": list(map(float, t.get("bbox", []))),
                    "attrs": t.get("attrs", {}),
                    "roi": roi_name
                }
                if self._meta_out:
                    rec.update(self._meta_out)
                self.track_sink.send(rec)
            self._last_track_emit = now

        # Summary: accumulate per frame, emit aggregated on interval
        if self.summary_enabled:
            # accumulate
            for k, v in per_frame_counts.items():
                try:
                    self._summary_accum[k] = self._summary_accum.get(
                        k, 0) + int(v)
                except Exception:
                    # ignore non-int convertible
                    pass
            self._summary_frames += 1
            if self._summary_first_ts_ms is None:
                self._summary_first_ts_ms = ts_ms

            if (now - self._last_summary_emit) >= self.summary_min_interval:
                if self._summary_accum:
                    # Compute averages across frames (mean per frame) for more realistic counts
                    duration_sec = max(
                        0.0, now - (self._last_summary_emit or now))
                    denom = max(1, self._summary_frames)
                    averaged = {k: (float(v) / denom)
                                for k, v in self._summary_accum.items()}
                    summary = {
                        "ts_ms": ts_ms,
                        "type": "summary",
                        "roi": roi_name,
                        "frames": self._summary_frames,
                        "duration_sec": duration_sec,
                        **averaged
                    }
                    if self._meta_out:
                        summary.update(self._meta_out)
                    self.summary_sink.send(summary)
                    # maintain rolling window for aggregates()
                    self.events.append((now, summary))
                    self._prune(now)
                # reset accumulators regardless
                self._summary_accum = {}
                self._summary_frames = 0
                self._summary_first_ts_ms = None
                self._last_summary_emit = now

    def aggregates(self) -> Dict[str, Any]:
        # rolling window aggregates/rates (per minute approximation)
        now = time.time()
        self._prune(now)
        # compute simple averages over window
        if not self.events:
            return {"window_sec": self.window}
        sums = {}
        for _, s in self.events:
            for k, v in s.items():
                if k in ("ts_ms", "type", "roi", "frames", "duration_sec"):
                    continue
                try:
                    sums[k] = sums.get(k, 0) + float(v)
                except Exception:
                    pass
        n = max(1, len(self.events))
        avgs = {f"avg_{k}": (v / n) for k, v in sums.items()}
        return {"window_sec": self.window, **avgs}

    def _prune(self, now: float = None):
        if now is None:
            now = time.time()
        while self.events and (now - self.events[0][0] > self.window):
            self.events.popleft()

    def close(self):
        # Close sinks
        seen = set()
        for s in (getattr(self, "track_sink", None), getattr(self, "summary_sink", None)):
            if s and id(s) not in seen:
                try:
                    s.close()
                except Exception:
                    pass
                seen.add(id(s))
