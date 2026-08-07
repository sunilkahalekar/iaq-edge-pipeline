"""
test_model.py — confirms the YOLO model loads and produces detections on
a real captured frame, independent of the camera loop and the database.
Run this SECOND, after test_camera.py has succeeded.

Usage:
    python3 test_model.py --model best_ncnn_model --image /home/pi4/iaq_data/camera_test.jpg --task segment
    python3 test_model.py --model best.pt --image /home/pi4/iaq_data/camera_test.jpg --task segment

By default also saves an annotated copy of the image (boxes/masks/labels
drawn on it) next to the input image, so you can pull it off the Pi and
actually look at what the model saw -- this is often faster than reading
a wall of printed detections.
"""

import argparse
import os
import time
import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model (.pt file or ncnn export folder)")
    ap.add_argument("--image", required=True, help="Path to a test image (e.g. from test_camera.py)")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--task", default=None, choices=[None, "detect", "segment"],
                     help="Explicitly set the model task if the export/model "
                          "metadata doesn't carry it correctly (check for a "
                          "'Unable to automatically guess model task' warning "
                          "-- if you see that, you MUST pass --task explicitly).")
    ap.add_argument("--save-annotated", default=None,
                     help="Path to save the annotated output image. Defaults "
                          "to '<image>_annotated.jpg' next to the input image. "
                          "Pass 'none' to skip saving.")
    args = ap.parse_args()

    print(f"Loading model from: {args.model}" + (f"  (task={args.task})" if args.task else ""))
    t0 = time.time()
    model = YOLO(args.model, task=args.task)
    print(f"Model loaded in {time.time()-t0:.2f}s -> resolved task: {model.task}")

    print("\nModel class names:")
    for cid, cname in sorted(model.names.items()):
        print(f"  [{cid:>3}] {cname}")

    expected = ["person"] + [f"machine_{m}_door_open" for m in (1, 2, 3)] \
        + [f"machine_{m}_door_closed" for m in (1, 2, 3)]
    missing = [c for c in expected if c not in model.names.values()]
    if missing:
        print(f"\nWARNING: these expected class names are missing from the model: {missing}")

    print(f"\nRunning inference on: {args.image}")
    t0 = time.time()
    results = model(args.image, imgsz=args.imgsz, conf=0.20, verbose=False)
    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.3f}s  (this is your per-frame cost on this Pi)")

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print("\nNo detections found in this frame.")
        print("  This is fine if the test image genuinely has no person/door in it —")
        print("  but if it should, check: camera framing, confidence threshold, model path.")
    else:
        print(f"\n{len(boxes)} detections:")
        if len(boxes) > 50:
            print(f"  WARNING: {len(boxes)} detections is far too many for a "
                  f"real scene. This usually means the model task is wrong "
                  f"(e.g. a segmentation model loaded/exported as 'detect') "
                  f"and the output tensor is being misparsed. Try re-running "
                  f"with an explicit --task, e.g. --task segment.")
        for box in boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            if cid not in model.names:
                print(f"  UNKNOWN class id={cid}  conf={conf:.3f}  "
                      f"(model only defines {len(model.names)} classes: "
                      f"0-{max(model.names)}) -- this confirms a task/export "
                      f"mismatch, not a camera or database issue.")
                continue
            cls_name = model.names[cid]
            print(f"  {cls_name:30s} conf={conf:.3f}")

    # ── Save an annotated copy so you can actually look at what the model saw ──
    if args.save_annotated != "none":
        if args.save_annotated:
            out_path = args.save_annotated
        else:
            base, ext = os.path.splitext(args.image)
            out_path = f"{base}_annotated{ext or '.jpg'}"

        # results[0].plot() draws boxes (and masks, for segmentation models)
        # with class labels + confidence directly onto a copy of the frame.
        annotated = results[0].plot()
        cv2.imwrite(out_path, annotated)
        print(f"\nAnnotated image saved to: {out_path}")
        print(f"  Pull it off the Pi to inspect, e.g.:")
        print(f"  scp pi4@<PI_IP>:{os.path.abspath(out_path)} .")


if __name__ == "__main__":
    main()
