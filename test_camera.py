"""
test_camera.py — confirms the camera source opens and delivers frames,
completely independent of YOLO/ultralytics. Run this FIRST.

Usage:
    python3 test_camera.py --source 0
    python3 test_camera.py --source "rtsp://192.168.1.50/stream"
"""

import argparse
import time
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="Camera index (e.g. 0) or RTSP URL")
    ap.add_argument("--seconds", type=float, default=5.0, help="How long to sample frames")
    ap.add_argument("--save", default="/home/pi4/iaq_data/camera_test.jpg")
    args = ap.parse_args()

    # Allow numeric camera indices passed as strings
    source = int(args.source) if args.source.isdigit() else args.source

    print(f"Opening source: {source!r}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("FAILED: cv2.VideoCapture could not open this source.")
        print("  - If using a USB webcam: check `ls /dev/video*` and try that index")
        print("  - If using an RTSP camera: check the URL works in VLC/ffplay first")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Opened OK. Reported resolution: {w}x{h}  reported fps: {fps:.1f}")

    frame_count = 0
    last_frame = None
    start = time.time()
    while time.time() - start < args.seconds:
        success, frame = cap.read()
        if not success:
            print("  WARNING: a frame read failed mid-stream")
            continue
        frame_count += 1
        last_frame = frame

    cap.release()

    elapsed = time.time() - start
    actual_fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"\nCaptured {frame_count} frames in {elapsed:.1f}s -> actual {actual_fps:.1f} fps")

    if last_frame is not None:
        cv2.imwrite(args.save, last_frame)
        print(f"Saved last frame to: {args.save}  (copy this to your PC to eyeball it)")
    else:
        print("FAILED: source opened but never delivered a usable frame.")


if __name__ == "__main__":
    main()
