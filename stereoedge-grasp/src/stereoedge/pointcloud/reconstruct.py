"""Disparity -> organised point cloud, table-plane fit, object segmentation, normals."""
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.spatial import cKDTree
from ..config import StereoRig


def disparity_to_cloud(disp, rig: StereoRig):
    """Organised (H, W, 3) cloud in the left-camera frame + validity mask."""
    H, W = disp.shape
    valid = np.isfinite(disp) & (disp > 1.0)
    d = np.where(valid, disp, 1.0)
    Z = rig.fx * rig.baseline / d
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    P = np.stack([(u - rig.cx) / rig.fx * Z, (v - rig.cy) / rig.fy * Z, Z], -1).astype(np.float32)
    P[~valid] = np.nan
    return P, valid


def fit_plane_ransac(points, iters=200, thresh=0.006, seed=0):
    """Plane n.p + d = 0 with n oriented toward the camera (camera is the origin, so d > 0)."""
    rng = np.random.default_rng(seed)
    pts = points[rng.choice(len(points), min(len(points), 6000), replace=False)]
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ a
        cnt = np.sum(np.abs(pts @ n + d) < thresh)
        if cnt > best_n:
            best, best_n = (n, d), cnt
    n, d = best
    inl = pts[np.abs(pts @ n + d) < thresh]            # least-squares refit on inliers
    c = inl.mean(0)
    n = np.linalg.svd(inl - c, full_matrices=False)[2][-1]
    d = -n @ c
    if d < 0:
        n, d = -n, -d
    return n.astype(np.float32), float(d)


@dataclass
class ObjectCluster:
    id: int
    mask: np.ndarray            # (H, W) bool
    points: np.ndarray          # (N, 3) camera frame
    heights: np.ndarray         # (N,) above table
    normals: np.ndarray = None  # (N, 3)
    curvature: np.ndarray = None

    @property
    def top_height(self):
        return float(np.percentile(self.heights, 90))

    @property
    def centroid(self):
        return self.points.mean(0)


def segment_objects(P, valid, plane, min_height=0.012, min_area=120):
    n, d = plane
    Pz = np.where(valid[..., None], P, 0)
    height = Pz @ n + d
    mask = valid & (height > min_height) & (height < 0.25)
    m8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m8 &= valid.astype(np.uint8)
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    clusters = []
    for i in range(1, n_lab):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        m = lab == i
        clusters.append(ObjectCluster(len(clusters), m, P[m], height[m]))
    return clusters, height


def estimate_normals(points, k=16):
    """PCA surface normals + curvature (surface variation), oriented toward the camera."""
    tree = cKDTree(points)
    _, idx = tree.query(points, k=min(k, len(points)))
    nb = points[idx] - points[idx].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / nb.shape[1]
    w, v = np.linalg.eigh(cov)
    nrm = v[:, :, 0]
    curv = w[:, 0] / np.maximum(w.sum(1), 1e-12)
    flip = np.einsum("ni,ni->n", nrm, -points) < 0
    nrm[flip] *= -1
    return nrm.astype(np.float32), curv.astype(np.float32)
