"""Procedural stereo scene generator so the project runs with no dataset and no hardware.

A top-down height-field scene (table + boxes + cylinder + sphere) is rendered from the left
camera; the right view is produced by z-buffered forward warping with the true disparity.
Ground-truth depth is returned so stereo accuracy can be measured.
"""
import numpy as np
import cv2
from .config import StereoRig


def _noise(rng, h, w, sigma):
    n = rng.standard_normal((h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    return n / (n.std() + 1e-6)


def _height_field(X, Y):
    """Return (height above table, label) for plane samples X, Y (camera-frame metres)."""
    h = np.zeros_like(X)
    lab = np.zeros(X.shape, np.int32)

    def box(cx, cy, sx, sy, yaw_deg, height, idx):
        nonlocal h, lab
        c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
        u = (X - cx) * c + (Y - cy) * s
        v = -(X - cx) * s + (Y - cy) * c
        m = (np.abs(u) < sx / 2) & (np.abs(v) < sy / 2)
        h = np.where(m, height, h); lab = np.where(m, idx, lab)

    def cyl(cx, cy, r, height, idx):
        nonlocal h, lab
        m = (X - cx) ** 2 + (Y - cy) ** 2 < r * r
        h = np.where(m, height, h); lab = np.where(m, idx, lab)

    def sphere(cx, cy, r, idx):
        nonlocal h, lab
        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        m = d2 < r * r
        top = r + np.sqrt(np.maximum(r * r - d2, 0))
        h = np.where(m, top, h); lab = np.where(m, idx, lab)

    box(-0.12, -0.05, 0.070, 0.040, 25, 0.050, 1)
    cyl(0.06, 0.07, 0.025, 0.080, 2)
    box(0.15, -0.08, 0.045, 0.045, -15, 0.040, 3)
    sphere(-0.06, 0.11, 0.030, 4)
    return h, lab


_RGB = {0: (0.62, 0.50, 0.38), 1: (0.85, 0.35, 0.25), 2: (0.30, 0.75, 0.40),
        3: (0.25, 0.55, 0.95), 4: (0.75, 0.40, 0.85)}
TINTS = {k: v[::-1] for k, v in _RGB.items()}   # stored as BGR for OpenCV


def _render(rig, rng_tex, cam_x, cam_height):
    """Ray-cast the height field from a camera displaced by cam_x (metres) along +X."""
    H, W = rig.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    # ray-march from the highest possible surface down to the table; first hit wins
    Z = np.full((H, W), cam_height, np.float32)
    hit = np.zeros((H, W), bool)
    for z in np.arange(cam_height - 0.09, cam_height + 1e-4, 0.001, dtype=np.float32):
        X = cam_x + (u - rig.cx) / rig.fx * z
        Y = (v - rig.cy) / rig.fy * z
        h, _ = _height_field(X, Y)
        new = (~hit) & (h >= cam_height - z - 1e-6) & (h > 0)
        Z[new] = cam_height - h[new]
        hit |= new
    X = cam_x + (u - rig.cx) / rig.fx * Z
    Y = (v - rig.cy) / rig.fy * Z
    h, lab = _height_field(X, Y)
    # world-anchored texture (1 mm / texel, 1.2 m square), sampled bilinearly
    mx = ((X + 0.6) * 1000).astype(np.float32)
    my = ((Y + 0.6) * 1000).astype(np.float32)
    tex = cv2.remap(rng_tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img = np.zeros((H, W, 3), np.float32)
    for k, bgr in TINTS.items():
        m = lab == k
        for c in range(3):
            img[..., c][m] = tex[m] * bgr[c] * 1.5
    return img, Z, lab


def make_scene(rig: StereoRig = None, seed: int = 0, cam_height: float = 0.55):
    rig = rig or StereoRig()
    rng = np.random.default_rng(seed)
    T = 1200
    tex = 0.55 + 0.16 * _noise(rng, T, T, 2.0) + 0.10 * _noise(rng, T, T, 1.0) \
        + 0.08 * _noise(rng, T, T, 8.0)
    tex = np.clip(tex, 0.05, 1.0).astype(np.float32)
    left, Zl, lab = _render(rig, tex, 0.0, cam_height)
    right, _, _ = _render(rig, tex, rig.baseline, cam_height)
    left = np.clip(left + rng.normal(0, 0.008, left.shape), 0, 1)
    right = np.clip(right * 0.97 + rng.normal(0, 0.008, right.shape), 0, 1)
    disp = rig.fx * rig.baseline / Zl
    return dict(left=(left * 255).astype(np.uint8), right=(right * 255).astype(np.uint8),
                depth_gt=Zl, disp_gt=disp.astype(np.float32), labels=lab)


# Ground truth for tests: label -> (name, centre XY in camera frame [m], nominal grasp width [m])
OBJECTS = {1: ("box 70x40x50 mm", (-0.12, -0.05), 0.040), 2: ("cylinder d50 h80 mm", (0.06, 0.07), 0.050),
           3: ("box 45x45x40 mm", (0.15, -0.08), 0.045), 4: ("sphere d60 mm", (-0.06, 0.11), 0.060)}
