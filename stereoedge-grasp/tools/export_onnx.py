#!/usr/bin/env python3
"""Regenerate the static-shape ONNX stereo model (models/stereo_costvolume.onnx)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from stereoedge.inference.onnx_model import build_stereo_onnx

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="models/stereo_costvolume.onnx")
ap.add_argument("--height", type=int, default=240); ap.add_argument("--width", type=int, default=320)
ap.add_argument("--num-disp", type=int, default=48)
a = ap.parse_args()
print("wrote", build_stereo_onnx(a.out, a.height, a.width, a.num_disp))
