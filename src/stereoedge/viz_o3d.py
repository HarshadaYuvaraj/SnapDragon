"""Optional interactive Open3D viewer (`pip install open3d`; x64 / Apple-silicon / Linux wheels).
Open3D has no official Windows-on-ARM64 wheel, so on a Snapdragon PC use the matplotlib
renderer (default) or run this viewer on a separate machine from the saved .ply."""
import numpy as np
from .viz import gripper_segments
from .config import GripperSpec


def show(res, left_bgr, rig, spec=None, save_ply=None):
    import open3d as o3d
    spec = spec or GripperSpec()
    T = rig.T_base_cam
    ok = np.isfinite(res.cloud).all(-1)
    P = res.cloud[ok] @ T[:3, :3].T + T[:3, 3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(left_bgr[ok][:, ::-1] / 255.0)
    geoms = [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)]
    if save_ply:
        o3d.io.write_point_cloud(save_ply, pcd)
    for k, g in enumerate(res.grasps):
        gb = g.transformed(T)
        segs = gripper_segments(gb.center, gb.R, gb.width, spec) if k == 0 else []
        if segs:
            pts = np.array([p for s in segs for p in s]); lines = [[2 * i, 2 * i + 1] for i in range(len(segs))]
            ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(lines))
            ls.paint_uniform_color([1, 0.15, 0.15]); geoms.append(ls)
        for c in gb.contacts:
            s = o3d.geometry.TriangleMesh.create_sphere(0.004); s.translate(c)
            s.paint_uniform_color([1, 0, 0]); geoms.append(s)
    o3d.visualization.draw_geometries(geoms, window_name="StereoEdge-Grasp")
