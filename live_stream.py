"""
live_stream.py — live camera feed with YOLO inference drawn on it,
viewable from any browser on your network (phone, laptop, etc.) without
anything attached to the Pi itself.

HOW IT WORKS
  - A background thread continuously reads frames from the camera, runs
    YOLO inference every INFERENCE_STRIDE frames (same idea as the main
    pipeline), draws boxes/masks with results[0].plot(), and stores the
    latest annotated JPEG in a shared, lock-protected buffer.
  - A tiny Flask web server serves that buffer as an MJPEG stream at
    "/video" — this is the same streaming format most IP cameras use,
    so any browser can just display it as an <img> tag, no plugins.

USAGE
    python3 live_stream.py --model /home/pi4/yolo/models/best_ncnn_model \
        --task segment --source 0 --port 8000

Then from ANY device on the same network (phone, laptop):
    http://<PI_IP_ADDRESS>:8000

Find the Pi's IP with:
    hostname -I
"""

import argparse
import threading
import time
from datetime import datetime
import cv2
from flask import Flask, Response
from ultralytics import YOLO

from db_writer import make_detection_writer


class LiveInferenceWorker:
    def __init__(self, source, model_path, task, imgsz, conf, stride, target_width,
                 db_path=None):
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source!r}")

        print(f"Loading model: {model_path}" + (f"  (task={task})" if task else ""))
        self.model = YOLO(model_path, task=task)
        print(f"Model loaded -> resolved task: {self.model.task}")

        self.imgsz = imgsz
        self.conf = conf
        self.stride = max(1, stride)
        self.target_width = target_width

        # PI: real-time detection logging. None if --db not passed, so this
        # script still works as a pure viewer with no database dependency.
        self.det_writer = None
        if db_path:
            self.det_writer = make_detection_writer(db_path)
            print(f"Logging detections to: {db_path}  (table: raw_detections)")

        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._frame_count = 0
        self._last_infer_ms = 0.0
        self._last_det_count = 0
        self._stop = False

        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join(timeout=5)
        self.cap.release()
        if self.det_writer is not None:
            self.det_writer.close()

    def get_latest_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def get_stats(self):
        with self._lock:
            return self._frame_count, self._last_infer_ms, self._last_det_count

    def _log_detections(self, results):
        """Write one row per detected box to raw_detections, timestamped now."""
        if self.det_writer is None:
            return
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return
        now = datetime.now()
        iso_ts = now.isoformat(timespec="milliseconds")
        unix_ms = int(now.timestamp() * 1000)
        for box in boxes:
            cid = int(box.cls[0])
            if cid not in self.model.names:
                continue  # defensive: skip anything outside the known class map
            cls_name = self.model.names[cid]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            self.det_writer.write({
                "Timestamp_ISO8601": iso_ts, "Unix_ms": unix_ms,
                "Frame_Number": self._frame_count,
                "Class_Name": cls_name, "Confidence": round(conf, 4),
                "X1": round(x1, 1), "Y1": round(y1, 1),
                "X2": round(x2, 1), "Y2": round(y2, 1),
            })

    def _run(self):
        last_boxes_plot = None
        while not self._stop:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.05)
                continue

            self._frame_count += 1
            run_inference = (self._frame_count % self.stride == 0)

            if run_inference:
                t0 = time.time()
                results = self.model(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
                self._last_infer_ms = (time.time() - t0) * 1000
                last_boxes_plot = results[0].plot()  # draws boxes/masks + labels

                boxes = results[0].boxes
                self._last_det_count = 0 if boxes is None else len(boxes)
                self._log_detections(results)

            display_frame = last_boxes_plot if last_boxes_plot is not None else frame

            # Optionally downscale for the browser stream — inference already
            # ran at full/imgsz resolution above; this only affects what's sent
            # over the network, keeping the stream light on Pi CPU + bandwidth.
            if self.target_width and display_frame.shape[1] > self.target_width:
                scale = self.target_width / display_frame.shape[1]
                new_h = int(display_frame.shape[0] * scale)
                display_frame = cv2.resize(display_frame, (self.target_width, new_h))

            # Small on-frame HUD so you can see it's alive, how fast inference
            # is, and how many detections were just logged to the DB
            hud = f"frame {self._frame_count}  infer {self._last_infer_ms:.0f}ms  dets {self._last_det_count}"
            cv2.putText(display_frame, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

            ok, buf = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()


def build_app(worker: LiveInferenceWorker):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return """
        <html><head><title>IAQ Live Inference</title></head>
        <body style="background:#111;color:#eee;font-family:sans-serif;text-align:center;">
          <h2>Live YOLO Inference</h2>
          <img src="/video" style="max-width:95%;border:2px solid #444;">
          <p id="stats"></p>
          <script>
            setInterval(() => {
              fetch('/stats').then(r => r.json()).then(d => {
                document.getElementById('stats').innerText =
                  `frames seen: ${d.frame_count}   last inference: ${d.infer_ms.toFixed(0)} ms   last detections: ${d.det_count}`;
              });
            }, 1000);
          </script>
        </body></html>
        """

    @app.route("/video")
    def video():
        def generate():
            while True:
                jpeg = worker.get_latest_jpeg()
                if jpeg is not None:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(0.05)   # ~20 fps cap on the stream itself
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/stats")
    def stats():
        frame_count, infer_ms, det_count = worker.get_stats()
        return {"frame_count": frame_count, "infer_ms": infer_ms, "det_count": det_count}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="Camera index or RTSP URL")
    ap.add_argument("--model", required=True, help="Path to model (.pt or ncnn export folder)")
    ap.add_argument("--task", default=None, choices=[None, "detect", "segment"],
                     help="Explicitly set model task -- required if the model's "
                          "own metadata doesn't carry it (see earlier NCNN export notes).")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--stride", type=int, default=2,
                     help="Run inference every Nth frame; frames in between "
                          "reuse the last annotated result (same idea as "
                          "INFERENCE_STRIDE in the main pipeline).")
    ap.add_argument("--stream-width", type=int, default=800,
                     help="Downscale the browser stream to this width to save "
                          "bandwidth/CPU. Does not affect inference resolution.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", default=None,
                     help="Path to SQLite DB file to log detections into in "
                          "real time (table: raw_detections). Omit to run as "
                          "a pure viewer with no database writes.")
    args = ap.parse_args()

    worker = LiveInferenceWorker(
        source=args.source, model_path=args.model, task=args.task,
        imgsz=args.imgsz, conf=args.conf, stride=args.stride,
        target_width=args.stream_width, db_path=args.db,
    )
    worker.start()

    app = build_app(worker)
    print(f"\nStreaming at: http://<PI_IP_ADDRESS>:{args.port}")
    print("Find the Pi's IP with: hostname -I")
    try:
        app.run(host="0.0.0.0", port=args.port, threaded=True)
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
