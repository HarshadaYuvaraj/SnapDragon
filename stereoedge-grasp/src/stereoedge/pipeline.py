"""End-to-end pipeline with per-stage timing."""
import time
from dataclasses import dataclass, field
import numpy as np
from .config import StereoRig, GripperSpec
from .pointcloud.reconstruct import (disparity_to_cloud, fit_plane_ransac, segment_objects,
                                     estimate_normals)
from .grasp.antipodal import synthesize_grasps


@dataclass
class Result:
    disp: np.ndarray
    cloud: np.ndarray
    valid: np.ndarray
    plane: tuple
    clusters: list
    grasps: list                      # best grasp per object, sorted by score
    timings_ms: dict = field(default_factory=dict)

    @property
    def best(self):
        return self.grasps[0] if self.grasps else None


class StereoGraspPipeline:
    def __init__(self, rig: StereoRig, backend, gripper: GripperSpec = None):
        self.rig, self.backend, self.gripper = rig, backend, gripper or GripperSpec()

    def run(self, left_bgr, right_bgr) -> Result:
        tm = {}
        t = time.perf_counter()

        def lap(name):
            nonlocal t
            now = time.perf_counter(); tm[name] = (now - t) * 1e3; t = now

        disp = self.backend.compute(left_bgr, right_bgr); lap("stereo_matching")
        P, valid = disparity_to_cloud(disp, self.rig); lap("point_cloud")
        if valid.sum() < 500:
            raise RuntimeError("Too few valid stereo matches; check rectification/texture.")
        plane = fit_plane_ransac(P[valid]); lap("plane_fit")
        clusters, _ = segment_objects(P, valid, plane); lap("segmentation")
        for c in clusters:
            c.normals, c.curvature = estimate_normals(c.points)
        lap("normals")
        scene = np.concatenate([c.points for c in clusters]) if clusters else np.zeros((0, 3))
        grasps = []
        for c in clusters:
            grasps += synthesize_grasps(c, scene, plane, self.gripper, top_k=1)
        grasps.sort(key=lambda g: -g.score); lap("grasp_synthesis")
        tm["total"] = sum(tm.values())
        return Result(disp, P, valid, plane, clusters, grasps, tm)
