"""
roi_calibrator.py — draw PERSON_ROI directly on a real frame from your
actual camera, at your actual capture resolution. This replaces guessing
a scale factor for a polygon drawn on unknown, different footage.

USAGE
    python3 roi_calibrator.py --source 0

Then open http://<PI_IP_ADDRESS>:8002 in a browser:
  - Click points on the image to build the polygon (in order, around the
    zone you want to count people in).
  - Click "Undo last point" to remove a mistake.
  - Click "Generate PERSON_ROI code" to get a ready-to-paste Python array,
    sized exactly to this camera's real resolution -- paste it directly
    over the PERSON_ROI definition in iaq_pipeline_pi.py.
"""

import argparse
import cv2
from flask import Flask, render_template_string, Response

FRAME = {"jpeg": None, "w": 0, "h": 0}


def grab_frame(source):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source {source!r}")
    success, frame = cap.read()
    cap.release()
    if not success:
        raise RuntimeError("Camera opened but did not return a frame")
    h, w = frame.shape[:2]
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    FRAME["jpeg"] = buf.tobytes()
    FRAME["w"], FRAME["h"] = w, h


PAGE = """
<!DOCTYPE html><html><head><meta charset="utf-8"><title>ROI Calibrator</title>
<style>
  body { background:#0f1115; color:#e8eaed; font-family:sans-serif; padding:20px; }
  #wrap { position: relative; display: inline-block; }
  #frame { display:block; max-width:100%; border:2px solid #2a2e38; }
  #overlay { position:absolute; top:0; left:0; cursor:crosshair; }
  button { background:#4ade80; border:none; color:#0f1115; font-weight:700;
           padding:8px 16px; border-radius:8px; margin:4px 6px 4px 0; cursor:pointer; }
  textarea { width:100%; height:140px; background:#1a1d24; color:#e8eaed;
             border:1px solid #2a2e38; border-radius:8px; padding:10px; font-family:monospace; }
</style></head>
<body>
  <h2>Click points around the zone you want to monitor</h2>
  <p style="color:#8a8f9b;">Frame size: {{w}} x {{h}} px — click in order around the polygon boundary.</p>
  <div id="wrap">
    <img id="frame" src="/frame.jpg" width="{{w}}" height="{{h}}">
    <canvas id="overlay" width="{{w}}" height="{{h}}"></canvas>
  </div>
  <div>
    <button onclick="undoPoint()">Undo last point</button>
    <button onclick="clearPoints()">Clear all</button>
    <button onclick="generateCode()">Generate PERSON_ROI code</button>
  </div>
  <textarea id="output" placeholder="Click 'Generate PERSON_ROI code' once you've drawn your polygon..."></textarea>

<script>
let points = [];
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = Math.round((e.clientX - rect.left) * scaleX);
  const y = Math.round((e.clientY - rect.top) * scaleY);
  points.push([x, y]);
  draw();
});

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#ff3c3c'; ctx.fillStyle = '#ff3c3c'; ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((p, i) => {
    if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
    ctx.fillRect(p[0]-3, p[1]-3, 6, 6);
  });
  if (points.length > 2) ctx.closePath();
  ctx.stroke();
}

function undoPoint() { points.pop(); draw(); }
function clearPoints() { points = []; draw(); }

function generateCode() {
  if (points.length < 3) {
    document.getElementById('output').value = 'Click at least 3 points first.';
    return;
  }
  const lines = points.map(p => `    [${p[0]}, ${p[1]}]`).join(',\\n');
  const code = `PERSON_ROI = np.array([\\n${lines}\\n])`;
  document.getElementById('output').value = code;
}
</script>
</body></html>
"""


def build_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(PAGE, w=FRAME["w"], h=FRAME["h"])

    @app.route("/frame.jpg")
    def frame_jpg():
        return Response(FRAME["jpeg"], mimetype="image/jpeg")

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--port", type=int, default=8002)
    args = ap.parse_args()

    print(f"Grabbing a frame from source={args.source!r}...")
    grab_frame(args.source)
    print(f"Frame captured: {FRAME['w']}x{FRAME['h']}")
    print(f"\nOpen: http://<PI_IP_ADDRESS>:{args.port}")

    app = build_app()
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
