"""Stereo rectification from an OpenCV calibration (.npz) for real camera pairs.

npz keys: K1, D1, K2, D2, R, T (right camera relative to left, OpenCV convention), size=(w, h)
Produce it with cv2.stereoCalibrate on a checkerboard set. Returns rectified images resized to
the network input size and a StereoRig with the matching rectified intrinsics.
"""
import numpy as np
import cv2
from ..config import StereoRig


class Rectifier:
    def __init__(self, calib_path, out_size=(320, 240), num_disp=48):
        c = np.load(calib_path)
        w, h = [int(v) for v in c["size"]]
        R1, R2, P1, P2, Q, *_ = cv2.stereoRectify(c["K1"], c["D1"], c["K2"], c["D2"], (w, h),
                                                  c["R"], c["T"], flags=cv2.CALIB_ZERO_DISPARITY,
                                                  alpha=0)
        self.m1 = cv2.initUndistortRectifyMap(c["K1"], c["D1"], R1, P1, (w, h), cv2.CV_16SC2)
        self.m2 = cv2.initUndistortRectifyMap(c["K2"], c["D2"], R2, P2, (w, h), cv2.CV_16SC2)
        s = out_size[0] / w
        self.out_size = out_size
        self.rig = StereoRig(width=out_size[0], height=out_size[1], fx=float(P1[0, 0] * s),
                             fy=float(P1[1, 1] * s), cx=float(P1[0, 2] * s),
                             cy=float(P1[1, 2] * s), baseline=float(-P2[0, 3] / P2[0, 0]),
                             num_disp=num_disp)

    def __call__(self, left, right):
        l = cv2.remap(left, *self.m1, cv2.INTER_LINEAR)
        r = cv2.remap(right, *self.m2, cv2.INTER_LINEAR)
        return (cv2.resize(l, self.out_size, interpolation=cv2.INTER_AREA),
                cv2.resize(r, self.out_size, interpolation=cv2.INTER_AREA))
