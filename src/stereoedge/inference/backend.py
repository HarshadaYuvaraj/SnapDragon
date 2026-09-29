"""Inference backends. One interface, three execution targets:

  sgbm      OpenCV Semi-Global Block Matching on CPU  (classical baseline, NOT an NPU workload)
  onnx-cpu  ONNX stereo network on onnxruntime CPUExecutionProvider
  onnx-qnn  same ONNX network on the Hexagon NPU through QNNExecutionProvider
            (needs `onnxruntime-qnn` on a Windows-on-ARM64 Snapdragon PC; falls back to CPU
            with a warning if the provider is not present)
"""
import os
import sys
import time
import warnings
import numpy as np
import cv2

from ..config import StereoRig


class StereoBackend:
    name = "base"

    def __init__(self, rig: StereoRig):
        self.rig = rig

    def compute(self, left_bgr, right_bgr):
        """Return float32 disparity (H, W) in pixels; NaN where invalid."""
        raise NotImplementedError


class SGBMBackend(StereoBackend):
    name = "sgbm-cpu"

    def __init__(self, rig):
        super().__init__(rig)
        bs = 5
        self.m = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=rig.num_disp, blockSize=bs,
            P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=1, uniquenessRatio=8,
            speckleWindowSize=60, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def compute(self, left_bgr, right_bgr):
        g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        d = self.m.compute(g(left_bgr), g(right_bgr)).astype(np.float32) / 16.0
        d[d < 1.0] = np.nan
        return d


def qnn_backend_path():
    return "QnnHtp.dll" if sys.platform.startswith("win") else "libQnnHtp.so"


def resolve_providers(target: str):
    """target: 'cpu' | 'qnn' | 'auto'. Returns (providers list, human label)."""
    import onnxruntime as ort
    avail = ort.get_available_providers()
    want_qnn = target in ("qnn", "auto") and "QNNExecutionProvider" in avail
    if target == "qnn" and not want_qnn:
        warnings.warn("QNNExecutionProvider not available (install `onnxruntime-qnn` on a "
                      "Snapdragon Windows-on-ARM64 PC). Falling back to CPU.")
    if want_qnn:
        opts = {"backend_path": qnn_backend_path(),
                "htp_performance_mode": "burst",
                "enable_htp_fp16_precision": "1"}
        return [("QNNExecutionProvider", opts), "CPUExecutionProvider"], "QNN-HTP (Hexagon NPU)"
    return ["CPUExecutionProvider"], "CPU"


class OnnxStereoBackend(StereoBackend):
    def __init__(self, rig, model_path, target="cpu"):
        super().__init__(rig)
        import onnxruntime as ort
        if not os.path.exists(model_path):
            from .onnx_model import build_stereo_onnx
            build_stereo_onnx(model_path, rig.height, rig.width, rig.num_disp)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        providers, label = resolve_providers(target)
        self.sess = ort.InferenceSession(model_path, so, providers=providers)
        self.active = self.sess.get_providers()[0]
        self.name = f"onnx-{'qnn' if 'QNN' in self.active else 'cpu'}"
        self.label = label

    @staticmethod
    def _prep(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return g[None, None]

    def compute(self, left_bgr, right_bgr):
        cost, disp = self.sess.run(["cost", "disparity"],
                                   {"left": self._prep(left_bgr), "right": self._prep(right_bgr)})
        cost, disp = cost[0], disp[0]
        D = cost.shape[0]
        idx = np.clip(disp.astype(np.int64), 1, D - 2)
        c0 = np.take_along_axis(cost, idx[None] - 1, 0)[0]
        c1 = np.take_along_axis(cost, idx[None], 0)[0]
        c2 = np.take_along_axis(cost, idx[None] + 1, 0)[0]
        den = c0 - 2 * c1 + c2
        ok = np.abs(den) > 1e-9
        sub = np.where(ok, 0.5 * (c0 - c2) / np.where(ok, den, 1.0), 0.0)
        out = idx + np.clip(sub, -0.5, 0.5)
        # validity: reject weak matches (low texture / occlusion) and disparity at range limits
        bad = (c1 > 0.06) | (disp < 2) | (disp > D - 3)
        out = out.astype(np.float32)
        out[bad] = np.nan
        return out


def make_backend(kind: str, rig: StereoRig, model_path="models/stereo_costvolume.onnx"):
    if kind == "sgbm":
        return SGBMBackend(rig)
    if kind in ("onnx-cpu", "cpu"):
        return OnnxStereoBackend(rig, model_path, "cpu")
    if kind in ("onnx-qnn", "qnn"):
        return OnnxStereoBackend(rig, model_path, "qnn")
    if kind == "auto":
        return OnnxStereoBackend(rig, model_path, "auto")
    raise ValueError(f"unknown backend {kind}")
