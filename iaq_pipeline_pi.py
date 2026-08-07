"""
IAQ Feature Extraction Pipeline  — Raspberry Pi 4 Edition
==========================================================

Adapted from the original desktop/GPU pipeline (v6). Changes vs. the
original are marked with "# PI:" comments throughout so you can diff
against your original file easily.

WHAT CHANGED AND WHY
  1. Input is a LIVE source (USB cam / Pi Cam / RTSP), not a folder of
     recorded video files. No tkinter, no filename timestamp parsing.
  2. Model is loaded from an NCNN export (or ONNX) for CPU speed on
     ARM. fp16 (`half=True`) is removed — it only helps on CUDA.
  3. Tracker switched from botsort.yaml -> bytetrack.yaml. Botsort's
     appearance/ReID branch is redundant CPU cost since this pipeline
     already runs its own StableIDRegistry for cross-occlusion ReID.
  4. Output goes to SQLite (db_writer.py) instead of three CSV files.
  5. Annotated-video writing is OFF by default (ANNOTATE=False) —
     encoding video while also running inference will starve a Pi 4.
     Turn it on only for short debugging sessions.
  6. Keyframe timing is wall-clock based (time.time()) rather than
     frame-count/fps based, since a live source's delivered fps can
     drift under CPU load in a way a recorded file's never does.

EVERYTHING ELSE (DoorDebouncer, StableIDRegistry, compute_Dt,
compute_Ht, WindowBuffer, class_colour/make_label/read_door_detections)
is carried over unchanged — that logic is hardware-agnostic.
"""

import time
import signal
import threading
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from datetime import datetime, timedelta
from collections import deque
from flask import Flask, Response

from db_writer import make_writers

# ══════════════════════════════ CONFIGURATION ══════════════════════════════

# PI: point this at an NCNN export directory (recommended) or an ONNX file.
#     Export once on your dev machine or on the Pi itself:
#       yolo export model=best.pt format=ncnn imgsz=480
#     -> produces a "best_ncnn_model" folder. Point MODEL_PATH at that folder.
MODEL_PATH = "/home/pi4/yolo/models/best_ncnn_model"

# PI: MUST match the task the model was actually trained/exported with.
#     This model was trained under runs/segment/... -> it's a segmentation
#     model. Loading it without this explicit task caused the earlier
#     "300 phantom detections" bug (NCNN export lost the task metadata,
#     ultralytics guessed 'detect', and misread mask coefficients as
#     bogus extra classes). Confirmed working value: "segment".
MODEL_TASK = "segment"

# PI: live source. Examples:
#   0                                      -> first USB webcam
#   "rtsp://user:pass@192.168.1.50/stream" -> IP camera
#   "/dev/video0"                          -> explicit v4l2 device
VIDEO_SOURCE = 0

# PI: SQLite file. Put this on a USB SSD if you expect sustained writes;
#     the SD card will wear out faster under frequent small writes.
DB_PATH = "/home/pi4/iaq_data/iaq.db"

# PI: how long a written row can sit uncommitted before it's flushed to
# disk and becomes visible to the dashboard. This is a direct trade-off:
#   - Lower value  -> near-instant dashboard updates, more frequent flash
#                     writes (more SD card wear over time)
#   - Higher value -> less flash wear, but detections can take up to this
#                     many seconds to appear anywhere outside this process
# 1-2s feels "instant" to a person watching the dashboard while still
# batching multiple rows per commit (far better than the ~34 rows/sec
# ceiling of committing on every single row). If you're on a USB SSD
# rather than a microSD card, wear is a non-issue -- go as low as you want.
DB_COMMIT_INTERVAL_SECONDS = 2.0

# PI: keep annotated-frame drawing/writing OFF in production. It's the
#     single most expensive thing this script can do on a Pi 4 CPU.
ANNOTATE = False
ANNOTATED_OUTPUT_PATH = "/home/pi4/iaq_data/annotated_live.mp4"

# PI: integrated live browser stream, running INSIDE this same process --
# same camera handle, same model instance, same detections that get
# written to the DB. This replaces running live_stream.py as a separate
# process, which was opening a SECOND competing connection to the same
# camera device and could starve or corrupt frames for both processes.
ENABLE_LIVE_STREAM = True
STREAM_PORT = 8000
STREAM_WIDTH = 640   # downscale for network/CPU; does not affect inference resolution

PERSON_ROI = np.array([
    [440, 269], [488, 255], [556, 239], [603, 225], [648, 222], [699, 210],
    [768, 196], [789, 188], [851, 207], [1092, 243], [1326, 365], [1624, 516],
    [1674, 542], [1470, 819], [1113, 1023], [856, 1060], [768, 1030],
    [520, 612], [451, 390], [428, 321], [422, 294]
])
# PI: if you drop inference resolution below the resolution these ROI points
# were drawn at, rescale them (see `scale_roi()` below) instead of hand-editing.

MACHINES = [1, 2, 3]
WINDOW_SECONDS = 60

# PI: count every detected person anywhere in frame, instead of only
# those inside the PERSON_ROI polygon. Set False to go back to
# ROI-gated counting (needs roi_calibrator.py run first).
COUNT_ENTIRE_FRAME = True

# PI: deterministic per-second sampling -- one capture+inference+DB-write
# cycle targeted at each wall-clock second boundary (13:05:01, 13:05:02, ...),
# instead of the old "run inference every Nth captured frame" approach.
# If inference genuinely takes longer than this on your hardware, the
# pipeline logs it honestly and resyncs to the next second rather than
# silently drifting or skipping seconds without telling you.
TARGET_TICK_SECONDS = 1.0
CAM_FLUSH_FRAMES = 0   # >0: discard N buffered frames before reading, for
                       # freshest possible frame -- costs extra decode time,
                       # only enable if the video looks laggy/stale
INFERENCE_IMGSZ = 480         # PI: try 480 first; drop to 384/320 if still too slow
TRACKER_CONFIG = "bytetrack.yaml"   # PI: lighter than botsort.yaml, no appearance model

PERSON_CONF = 0.25
DOOR_CONF = 0.40
DOOR_INFERENCE_CONF = 0.20

DOOR_CONF_PER_MACHINE = {1: 0.40, 2: 0.40, 3: 0.25}
DOOR_HOLD_PER_MACHINE = {1: 2, 2: 2, 3: 1}

M3_CROP = None   # (x1, y1, x2, y2) in INFERENCE_IMGSZ-scaled coordinates, if used

DOOR_VOTE_WINDOW = 3
DOOR_HOLD_SECONDS = 2

LOST_TRACK_TTL = 90
REID_IOU_THRESHOLD = 0.35
REID_COSINE_THRESHOLD = 0.60
REID_CROP_SIZE = (64, 128)

FARNEBACK_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
)

CLR_PERSON = sv.Color(r=100, g=230, b=100)
CLR_DOOR_OPN = sv.Color(r=255, g=190, b=70)
CLR_DOOR_CLS = sv.Color(r=130, g=200, b=255)
CLR_OTHER = sv.Color(r=200, g=200, b=200)
CLR_ROI = sv.Color(r=255, g=60, b=60)

DIAG_MODE = True   # PI: leave on while tuning thresholds on real Pi footage; turn off in production


def _sv_default_palette() -> sv.ColorPalette:
    if hasattr(sv.ColorPalette, "DEFAULT"):
        return sv.ColorPalette.DEFAULT
    return sv.ColorPalette.default()


def scale_roi(roi: np.ndarray, src_wh: tuple, dst_wh: tuple) -> np.ndarray:
    """Rescale ROI polygon points if you capture/infer at a different
    resolution than the one the polygon was originally drawn on."""
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    return (roi * np.array([sx, sy])).astype(int)


# ══════════════════════════════ DOOR DEBOUNCER (unchanged) ═══════════════════

class DoorDebouncer:
    def __init__(self, initial_state: str = "Closed", hold_seconds: int = None):
        self._hold = hold_seconds if hold_seconds is not None else DOOR_HOLD_SECONDS
        self._committed = initial_state
        self._pending = None
        self._pending_count = 0
        self._vote_buf = deque(maxlen=DOOR_VOTE_WINDOW)
        for _ in range(DOOR_VOTE_WINDOW):
            self._vote_buf.append(initial_state)

    def observe(self, raw):
        vote = raw if raw is not None else self._committed
        self._vote_buf.append(vote)
        open_count = sum(1 for v in self._vote_buf if v == "Open")
        majority = "Open" if open_count > len(self._vote_buf) - open_count else "Closed"
        transition = False
        if majority != self._committed:
            if majority == self._pending:
                self._pending_count += 1
            else:
                self._pending = majority
                self._pending_count = 1
            if self._pending_count >= self._hold:
                self._committed = majority
                self._pending = None
                self._pending_count = 0
                transition = True
        else:
            self._pending = None
            self._pending_count = 0
        return self._committed, transition

    @property
    def state(self):
        return self._committed

    def reset(self, state: str = "Closed"):
        self.__init__(state, hold_seconds=self._hold)


# ══════════════════════════════ STABLE ReID REGISTRY (unchanged) ═════════════

class TrackState:
    __slots__ = ("global_id", "last_xyxy", "last_crop_feat", "frames_lost", "in_zone")

    def __init__(self, gid, xyxy, feat, in_zone=False):
        self.global_id = gid
        self.last_xyxy = xyxy
        self.last_crop_feat = feat
        self.frames_lost = 0
        self.in_zone = in_zone


def _extract_crop_feature(frame: np.ndarray, xyxy) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = max(0, int(xyxy[0])); y1 = max(0, int(xyxy[1]))
    x2 = min(w, int(xyxy[2])); y2 = min(h, int(xyxy[3]))
    if x2 <= x1 or y2 <= y1:
        return np.zeros(96, dtype=np.float32)
    crop = cv2.resize(frame[y1:y2, x1:x2], REID_CROP_SIZE)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    feat = np.concatenate([hist_h, hist_s, hist_v])
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat


def _iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


class StableIDRegistry:
    def __init__(self):
        self._next_gid = 1
        self._active = {}
        self._lost = {}
        self._tid_to_gid = {}

    def update(self, frame, tracker_ids, xyxys, in_zone_mask) -> dict:
        for st in list(self._active.values()):
            st.frames_lost += 1
        for tid in [t for t, st in self._active.items() if st.frames_lost > LOST_TRACK_TTL]:
            self._lost[tid] = self._active.pop(tid)
        for tid in [t for t, st in self._lost.items() if st.frames_lost > LOST_TRACK_TTL * 3]:
            del self._lost[tid]

        result = {}
        for i, tid in enumerate(tracker_ids):
            tid = int(tid)
            xyxy = xyxys[i]
            in_roi = bool(in_zone_mask[i])
            feat = _extract_crop_feature(frame, xyxy)

            if tid in self._active:
                st = self._active[tid]
                st.last_xyxy = xyxy; st.last_crop_feat = feat
                st.frames_lost = 0; st.in_zone = in_roi
                result[tid] = st.global_id
                continue

            best_tid, best_score = None, -1.0
            for lt_tid, lt_st in self._lost.items():
                iou = _iou(xyxy, lt_st.last_xyxy)
                if iou < REID_IOU_THRESHOLD:
                    continue
                cos = float(np.dot(feat, lt_st.last_crop_feat))
                score = 0.5 * iou + 0.5 * cos
                if score > best_score:
                    best_score = score; best_tid = lt_tid

            threshold = 0.5 * (REID_IOU_THRESHOLD + REID_COSINE_THRESHOLD)
            if best_tid is not None and best_score >= threshold:
                recovered = self._lost.pop(best_tid)
                recovered.last_xyxy = xyxy; recovered.last_crop_feat = feat
                recovered.frames_lost = 0; recovered.in_zone = in_roi
                self._active[tid] = recovered
                self._tid_to_gid[tid] = recovered.global_id
                result[tid] = recovered.global_id
            else:
                gid = self._next_gid; self._next_gid += 1
                st = TrackState(gid, xyxy, feat, in_roi)
                self._active[tid] = st
                self._tid_to_gid[tid] = gid
                result[tid] = gid

        return result


# ══════════════════════════════ D_t / H_t COMPUTATION (unchanged) ════════════

def compute_Dt(seq: list) -> dict:
    T = len(seq) or 1
    tau_open = int(sum(seq))
    f_trans = int(sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1]))
    rho_open = round(tau_open / T, 4)
    eps_max = run = 0
    for s in seq:
        run = run + 1 if s == 1 else 0
        eps_max = max(eps_max, run)
    first_open = next((i for i, s in enumerate(seq) if s == 1), None)
    phi_open = round(first_open / T, 4) if first_open is not None else 1.0
    return dict(tau_open=tau_open, f_trans=f_trans,
                rho_open=rho_open, eps_max=eps_max, phi_open=phi_open)


def compute_frame_motion(fu, fv, boxes: list) -> float:
    if not boxes:
        return 0.0
    h, w = fu.shape
    mag = np.sqrt(fu ** 2 + fv ** 2)
    scores = []
    for (x1, y1, x2, y2) in boxes:
        r = mag[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if r.size:
            scores.append(float(r.mean()))
    return float(np.mean(scores)) if scores else 0.0


def compute_Ht(motion_seq: list, n_max: int) -> dict:
    arr = np.array(motion_seq, dtype=float)
    return dict(
        n_person=n_max,
        mu_motion=round(float(arr.mean()) if arr.size else 0.0, 6),
        sigma2_motion=round(float(arr.var()) if arr.size else 0.0, 6),
    )


class WindowBuffer:
    def __init__(self, ws=WINDOW_SECONDS):
        self.ws = ws
        self._reset()

    def _reset(self):
        self.door_seq = {m: [] for m in MACHINES}
        self.motion_seq = []
        self.person_counts = []
        self.count = 0
        self.window_start_ts = None

    def push(self, ts, door_binary, motion, n_person):
        if self.count == 0:
            self.window_start_ts = ts
        for m in MACHINES:
            self.door_seq[m].append(door_binary[m])
        self.motion_seq.append(motion)
        self.person_counts.append(n_person)
        self.count += 1

    def is_full(self):
        return self.count >= self.ws

    def flush(self) -> dict:
        if self.count == 0:
            return {}
        row = {"window_start": self.window_start_ts.strftime("%Y-%m-%d %H:%M:%S")}
        for m in MACHINES:
            for k, v in compute_Dt(self.door_seq[m]).items():
                row[f"M{m}_{k}"] = v
        row.update(compute_Ht(self.motion_seq, max(self.person_counts, default=0)))
        self._reset()
        return row


# ══════════════════════════════ HELPERS (unchanged) ═══════════════════════════

def class_colour(class_name: str) -> sv.Color:
    if class_name == "person":
        return CLR_PERSON
    if "open" in class_name:
        return CLR_DOOR_OPN
    if "closed" in class_name:
        return CLR_DOOR_CLS
    return CLR_OTHER


def make_label(class_name: str, global_id) -> str:
    if class_name == "person":
        gid = f"#{int(global_id)}" if global_id is not None else "#?"
        return f"{gid}  Person"
    for m in MACHINES:
        if f"machine_{m}_door_open" in class_name:
            return f"M{m}  Open"
        if f"machine_{m}_door_closed" in class_name:
            return f"M{m}  Closed"
    return class_name


def read_door_detections(results, model_names: dict, extra_results=None) -> tuple:
    raw_state = {m: None for m in MACHINES}
    raw_conf = {m: None for m in MACHINES}
    all_door_hits = []

    def _process(res):
        if res.boxes is None:
            return
        for box in res.boxes:
            conf = float(box.conf[0])
            lbl = model_names[int(box.cls[0])]
            is_door = any(
                lbl == f"machine_{m}_door_open" or lbl == f"machine_{m}_door_closed"
                for m in MACHINES
            )
            if is_door:
                all_door_hits.append((lbl, round(conf, 3)))
            for m in MACHINES:
                per_m_conf = DOOR_CONF_PER_MACHINE.get(m, DOOR_CONF)
                if conf < per_m_conf:
                    continue
                if lbl == f"machine_{m}_door_open":
                    if raw_conf[m] is None or conf > raw_conf[m]:
                        raw_state[m] = "Open"; raw_conf[m] = conf
                elif lbl == f"machine_{m}_door_closed":
                    if raw_conf[m] is None or conf > raw_conf[m]:
                        raw_state[m] = "Closed"; raw_conf[m] = conf

    _process(results)
    if extra_results is not None:
        _process(extra_results)
    return raw_state, raw_conf, all_door_hits


# ══════════════════════════════ INTEGRATED LIVE STREAM ═══════════════════════

class StreamBuffer:
    """
    Lock-protected holder for the latest annotated JPEG. The main
    inference loop pushes into this after every processed frame; a
    background Flask thread serves whatever is currently here. No
    second camera or model instance involved.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None

    def set(self, jpeg_bytes: bytes):
        with self._lock:
            self._jpeg = jpeg_bytes

    def get(self):
        with self._lock:
            return self._jpeg


def start_stream_server(buffer: StreamBuffer, port: int):
    app = Flask(__name__)

    @app.route("/video")
    def video():
        def generate():
            while True:
                jpeg = buffer.get()
                if jpeg is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(0.05)
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/")
    def index():
        return ('<html><body style="background:#111;margin:0;">'
                '<img src="/video" style="width:100%;"></body></html>')

    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False),
        daemon=True,
    )
    t.start()
    print(f"Live stream running at: http://<PI_IP_ADDRESS>:{port}  (embed this in the dashboard)")


# ══════════════════════════════ MAIN (live-capture, Pi-adapted) ═══════════════

_STOP = False


def _handle_sigterm(signum, frame):
    global _STOP
    _STOP = True


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    model = YOLO(MODEL_PATH, task=MODEL_TASK)
    name_to_id = {v: k for k, v in model.names.items()}

    print("\n  Model class names:")
    for cid, cname in sorted(model.names.items()):
        print(f"    [{cid:>3}] {cname}")

    expected_classes = (
        ["person"]
        + [f"machine_{m}_door_open" for m in MACHINES]
        + [f"machine_{m}_door_closed" for m in MACHINES]
    )
    missing = [c for c in expected_classes if c not in name_to_id]
    if missing:
        print("\n  WARNING — missing expected classes:", missing)

    target_ids = [name_to_id[c] for c in expected_classes if c in name_to_id]
    person_cid = name_to_id.get("person")

    zone = sv.PolygonZone(polygon=PERSON_ROI)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=CLR_ROI, thickness=2)

    global_door_state = {m: "Closed" for m in MACHINES}
    toggle_counts = {m: 0 for m in MACHINES}
    cum_setup_time = {m: 0.0 for m in MACHINES}
    cum_swap_time = {m: 0.0 for m in MACHINES}
    unique_global_ids = set()
    total_person_seconds = 0.0

    debouncers = {
        m: DoorDebouncer("Closed", hold_seconds=DOOR_HOLD_PER_MACHINE.get(m, DOOR_HOLD_SECONDS))
        for m in MACHINES
    }
    reid_registry = StableIDRegistry()

    # PI: one shared connection/thread serves all four tables now --
    # see db_writer.py's AsyncDBWriter docstring for why (avoids the
    # multi-connection "database is locked" issue).
    ct_writer, sec_writer, tel_writer, det_writer, shared_db_writer = make_writers(
        DB_PATH, commit_every_sec=DB_COMMIT_INTERVAL_SECONDS)
    window_buf = WindowBuffer(WINDOW_SECONDS)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"ERROR: could not open video source {VIDEO_SOURCE!r}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    out = None
    if ANNOTATE:
        out = cv2.VideoWriter(
            ANNOTATED_OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"),
            10.0, (frame_w, frame_h),   # PI: assume ~10fps effective throughput; adjust after measuring
        )

    print(f"\n🚀  Live pipeline started on source={VIDEO_SOURCE!r}  "
          f"({frame_w}x{frame_h})  imgsz={INFERENCE_IMGSZ}  "
          f"target_tick={TARGET_TICK_SECONDS}s")

    stream_buffer = None
    if ENABLE_LIVE_STREAM:
        stream_buffer = StreamBuffer()
        start_stream_server(stream_buffer, STREAM_PORT)
        print(f"    NOTE: PERSON_ROI was drawn at some earlier resolution -- "
              f"if it doesn't match {frame_w}x{frame_h}, zone/person counts "
              f"will be wrong. Use roi_calibrator.py to redraw it correctly "
              f"for this exact camera if you haven't already.")

    prev_gray = None
    frame_count = 0
    last_raw_state = {m: "Closed" for m in MACHINES}
    last_tick_start = None

    next_tick = time.monotonic()

    try:
        while not _STOP:
            now_mono = time.monotonic()
            if now_mono < next_tick:
                time.sleep(next_tick - now_mono)
            tick_start = time.monotonic()

            # PI: real elapsed time since the previous tick -- used for
            # accurate cumulative open/closed seconds and person-seconds,
            # instead of always assuming exactly 1.0s passed (which would
            # be wrong on any tick that ran over schedule).
            real_delta = (tick_start - last_tick_start) if last_tick_start else TARGET_TICK_SECONDS
            last_tick_start = tick_start

            # PI: authoritative wall-clock timestamp for this tick -- what
            # actually gets written to the DB. If the pipeline falls behind
            # schedule, this timestamp honestly reflects that (e.g. jumping
            # from 13:05:01 to 13:05:04) rather than faking evenly-spaced
            # seconds that were never actually sampled.
            now_dt = datetime.now()
            iso_ts = now_dt.isoformat(timespec="milliseconds")
            unix_ms = int(now_dt.timestamp() * 1000)

            for _ in range(CAM_FLUSH_FRAMES):
                cap.grab()
            success, frame = cap.read()
            if not success:
                print("  Frame read failed — retrying...")
                next_tick = time.monotonic() + TARGET_TICK_SECONDS
                continue

            results = model.track(
                frame, conf=DOOR_INFERENCE_CONF, verbose=False,
                classes=target_ids, persist=True,
                tracker=TRACKER_CONFIG, imgsz=INFERENCE_IMGSZ,
            )
            detections = sv.Detections.from_ultralytics(results[0])

            # One row per detected class, this exact tick's timestamp.
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cid = int(box.cls[0])
                    cls_name = model.names.get(cid, f"unknown_{cid}")
                    xyxy = box.xyxy[0].tolist()
                    det_writer.write({
                        "Timestamp_ISO8601": iso_ts,
                        "Unix_ms": unix_ms,
                        "Frame_Number": frame_count,
                        "Class_Name": cls_name,
                        "Confidence": round(float(box.conf[0]), 4),
                        "X1": round(xyxy[0], 1), "Y1": round(xyxy[1], 1),
                        "X2": round(xyxy[2], 1), "Y2": round(xyxy[3], 1),
                    })

            p_mask = (
                (detections.class_id == person_cid) & (detections.confidence >= PERSON_CONF)
                if person_cid is not None and detections.confidence is not None
                else (detections.class_id == person_cid)
                if person_cid is not None
                else np.zeros(len(detections), bool)
            )
            p_dets = detections[p_mask]
            if COUNT_ENTIRE_FRAME:
                roi_mask = np.ones(len(p_dets), dtype=bool)   # every person counts, no polygon gate
            else:
                roi_mask = zone.trigger(detections=p_dets)
            count_zone = int(roi_mask.sum())

            if DIAG_MODE:
                print(f"  [tick {now_dt.strftime('%H:%M:%S')}] "
                      f"persons_detected={len(p_dets)}  in_zone={count_zone}")

            global_map = {}
            if p_dets.tracker_id is not None and len(p_dets.tracker_id) > 0:
                global_map = reid_registry.update(frame, p_dets.tracker_id, p_dets.xyxy, roi_mask)

            frame_tracker_ids, frame_global_ids, person_boxes_kf = [], [], []
            if p_dets.tracker_id is not None:
                for i, (in_roi, tid) in enumerate(zip(roi_mask, p_dets.tracker_id)):
                    if in_roi:
                        tid_i = int(tid)
                        gid_i = global_map.get(tid_i, tid_i)
                        frame_tracker_ids.append(tid_i)
                        frame_global_ids.append(gid_i)
                        unique_global_ids.add(gid_i)
                        if p_dets.xyxy is not None:
                            x1, y1, x2, y2 = p_dets.xyxy[i].astype(int)
                            person_boxes_kf.append((x1, y1, x2, y2))

            total_person_seconds += count_zone * real_delta

            raw_door, raw_conf, all_door_hits = read_door_detections(results[0], model.names)

            debounced_state = {}
            for m in MACHINES:
                committed, changed = debouncers[m].observe(raw_door[m])
                debounced_state[m] = committed
                if changed:
                    toggle_counts[m] += 1
                    global_door_state[m] = committed
                if global_door_state[m] == "Closed":
                    cum_setup_time[m] += real_delta
                else:
                    cum_swap_time[m] += real_delta

            last_raw_state = {
                m: (raw_door[m] if raw_door[m] else last_raw_state.get(m, "Closed"))
                for m in MACHINES
            }

            # Every tick writes BOTH tables now -- no more separate
            # "captured frame" vs "keyframe" distinction, which used to
            # let Zone_Count/telemetry rows go stale between the two.
            tel_writer.write({
                "Timestamp_ISO8601": iso_ts, "Unix_ms": unix_ms,
                "M1_State_Debounced": debounced_state[1], "M2_State_Debounced": debounced_state[2],
                "M3_State_Debounced": debounced_state[3],
                "M1_State_Raw": last_raw_state[1], "M2_State_Raw": last_raw_state[2],
                "M3_State_Raw": last_raw_state[3],
                "Zone_Count": count_zone,
                "Tracked_Person_IDs": frame_tracker_ids, "Global_Person_IDs": frame_global_ids,
            })

            gray_kf = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_score = 0.0
            if prev_gray is not None and person_boxes_kf:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_kf, None, **FARNEBACK_PARAMS)
                motion_score = compute_frame_motion(flow[..., 0], flow[..., 1], person_boxes_kf)
            prev_gray = gray_kf

            sec_row = {
                "Timestamp": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "Person_Count_Max": count_zone,
                "Unique_Total_Persons": len(unique_global_ids),
                "Motion_Score_This_Second": round(motion_score, 6),
            }
            for m in MACHINES:
                sec_row[f"M{m}_State"] = global_door_state[m]
                sec_row[f"M{m}_Toggles"] = toggle_counts[m]
                sec_row[f"M{m}_Setup_Sec"] = round(cum_setup_time[m], 2)
                sec_row[f"M{m}_Swap_Sec"] = round(cum_swap_time[m], 2)
                sec_row[f"M{m}_Raw_State"] = last_raw_state[m]
            sec_writer.write(sec_row)

            door_binary_kf = {m: (1 if global_door_state[m] == "Open" else 0) for m in MACHINES}
            window_buf.push(ts=now_dt, door_binary=door_binary_kf,
                             motion=motion_score, n_person=count_zone)

            if window_buf.is_full():
                ct_row = window_buf.flush()
                if ct_row:
                    ct_writer.write(ct_row)
                    print(f"  [C_t] {ct_row['window_start']}  "
                          f"M1 τ={ct_row['M1_tau_open']:>2}s f={ct_row['M1_f_trans']:>2}  "
                          f"μ={ct_row['mu_motion']:.4f}  n={ct_row['n_person']}")

            if ANNOTATE or ENABLE_LIVE_STREAM:
                annotated = _annotate_frame(
                    frame, detections, global_map, zone_annotator,
                    debounced_state, last_raw_state, iso_ts, count_zone,
                    frame_global_ids, model, frame_w, True,
                )
                if ANNOTATE and out is not None:
                    out.write(annotated)
                if stream_buffer is not None:
                    display = annotated
                    if STREAM_WIDTH and display.shape[1] > STREAM_WIDTH:
                        scale = STREAM_WIDTH / display.shape[1]
                        display = cv2.resize(display, (STREAM_WIDTH, int(display.shape[0] * scale)))
                    ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        stream_buffer.set(buf.tobytes())

            frame_count += 1

            # ── Schedule the next tick ──
            processing_time = time.monotonic() - tick_start
            if processing_time > TARGET_TICK_SECONDS:
                behind_by = processing_time - TARGET_TICK_SECONDS
                if DIAG_MODE:
                    print(f"    ⚠ tick took {processing_time:.2f}s "
                          f"(target {TARGET_TICK_SECONDS}s) -- {behind_by:.2f}s behind. "
                          f"Resyncing to next tick rather than trying to catch up.")
                next_tick = time.monotonic()   # don't burst-process to catch up
            else:
                next_tick = tick_start + TARGET_TICK_SECONDS

    finally:
        cap.release()
        if out is not None:
            out.release()
        if window_buf.count > 0:
            ct_row = window_buf.flush()
            if ct_row:
                ct_writer.write(ct_row)
        # PI: close the ONE shared writer -- the per-table .close() calls
        # are intentional no-ops (see TableWriterHandle in db_writer.py)
        shared_db_writer.close()

    person_hours = total_person_seconds / 3600
    print(f"\n{'='*55}")
    print("📊  SESSION SUMMARY")
    print(f"  Unique people (ReID) : {len(unique_global_ids)}")
    print(f"  Person-hours in zone : {person_hours:.6f}")
    print(f"  DB file              : {DB_PATH}")
    print(f"{'='*55}")


def _annotate_frame(frame, detections, global_map, zone_annotator,
                     debounced_state, raw_state, iso_ts, count_zone,
                     frame_global_ids, model, frame_w, is_keyframe):
    if len(detections):
        colours = sv.ColorPalette(
            colors=[class_colour(model.names[int(cid)]) for cid in detections.class_id]
        )
    else:
        colours = _sv_default_palette()

    labels = []
    if detections.tracker_id is not None:
        for cid, tid in zip(detections.class_id, detections.tracker_id):
            cname = model.names[int(cid)]
            gid = (global_map.get(int(tid), int(tid)) if cname == "person" and tid is not None else None)
            labels.append(make_label(cname, gid))
    else:
        for cid in detections.class_id:
            labels.append(model.names[int(cid)])

    annotated = frame.copy() if COUNT_ENTIRE_FRAME else zone_annotator.annotate(scene=frame.copy())
    box_ann = sv.BoxAnnotator(color=colours)
    annotated = box_ann.annotate(scene=annotated, detections=detections)
    lbl_ann = sv.LabelAnnotator(color=colours, text_color=sv.Color.BLACK,
                                 text_scale=0.52, text_thickness=1, text_padding=4)
    annotated = lbl_ann.annotate(scene=annotated, detections=detections, labels=labels)

    hud1 = f"{iso_ts}  Zone:{count_zone}p  GIDs:{frame_global_ids}"
    hud2 = f"DOOR debounced -> M1:{debounced_state[1]}  M2:{debounced_state[2]}  M3:{debounced_state[3]}"
    hud3 = f"DOOR raw -> M1:{raw_state[1]}  M2:{raw_state[2]}  M3:{raw_state[3]}"
    for li, (hud, clr) in enumerate([(hud1, (220, 235, 220)), (hud2, (120, 220, 255)), (hud3, (180, 180, 100))]):
        (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        y_base = 10 + li * 24
        ov = annotated.copy()
        cv2.rectangle(ov, (4, y_base - 2), (min(tw + 12, frame_w - 4), y_base + th + 4), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.45, annotated, 0.55, 0, annotated)
        cv2.putText(annotated, hud, (8, y_base + th), cv2.FONT_HERSHEY_SIMPLEX, 0.46, clr, 1, cv2.LINE_AA)
    if is_keyframe:
        cv2.rectangle(annotated, (0, 0), (frame_w, 3), (80, 230, 80), -1)
    return annotated


if __name__ == "__main__":
    main()
