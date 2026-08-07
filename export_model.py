"""
export_model.py
================
Run this ONCE (either on the Pi itself, or on your dev machine and then
copy the output folder over) to convert your trained .pt weights into
an NCNN model — the fastest CPU inference backend Ultralytics supports
on ARM boards like the Pi 4.

Usage:
    python export_model.py --weights best.pt --imgsz 480

Output:
    A folder named "<weights-stem>_ncnn_model" next to your weights file.
    Point MODEL_PATH in iaq_pipeline_pi.py at that folder.

Notes:
  - Exporting itself can be done on the Pi, but it's slow there (limited
    RAM/CPU) and works fine cross-platform — exporting on a laptop and
    copying the resulting folder to the Pi is usually faster.
  - `imgsz` here should match INFERENCE_IMGSZ in iaq_pipeline_pi.py.
  - If NCNN export isn't available in your ultralytics version, ONNX is
    the fallback (`format=onnx`), run via onnxruntime on the Pi — slower
    than NCNN but still notably faster than raw PyTorch on ARM CPU.
"""

import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to trained best.pt")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--format", default="ncnn", choices=["ncnn", "onnx"])
    ap.add_argument("--task", default=None, choices=[None, "detect", "segment"],
                     help="Explicitly set the model task. If omitted, reads "
                          "it from the .pt file's own metadata and prints it "
                          "so you can confirm it's correct BEFORE exporting.")
    args = ap.parse_args()

    model = YOLO(args.weights, task=args.task)
    print(f"Model task: {model.task}")
    if model.task == "segment":
        print("  -> This is a segmentation model. The NCNN export must carry")
        print("     this task forward correctly, or downstream inference will")
        print("     misparse mask coefficients as bogus extra classes.")

    out_path = model.export(format=args.format, imgsz=args.imgsz, task=args.task or model.task)
    print(f"\nExported model to: {out_path}")
    print(f"IMPORTANT: when loading this exported model later, pass the same")
    print(f"task explicitly:  YOLO({out_path!r}, task={model.task!r})")
    print("Copy this to the Pi and point MODEL_PATH at it in iaq_pipeline_pi.py")


if __name__ == "__main__":
    main()
