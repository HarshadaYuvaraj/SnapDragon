#!/usr/bin/env python3
"""StereoEdge-Grasp end-to-end demo.

  python demo.py                              # synthetic scene, SGBM baseline
  python demo.py --backend onnx-cpu           # ONNX stereo network on CPU
  python demo.py --backend onnx-qnn           # ONNX network on the Snapdragon Hexagon NPU
  python demo.py --left L.png --right R.png --calib calib.npz --handeye T_base_cam.npy
Options: --gif (kinematic grasp animation)  --o3d (Open3D viewer)  --pybullet [--gui]
"""
import argparse, json, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from stereoedge.config import StereoRig, GripperSpec
from stereoedge.inference.backend import make_backend
from stereoedge.pipeline import StereoGraspPipeline
from stereoedge import viz


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--backend", default="sgbm", choices=["sgbm", "onnx-cpu", "onnx-qnn", "auto"])
    ap.add_argument("--model", default="models/stereo_costvolume.onnx")
    ap.add_argument("--left"); ap.add_argument("--right")
    ap.add_argument("--calib", help="stereoCalibrate .npz (K1,D1,K2,D2,R,T,size)")
    ap.add_argument("--handeye", help="4x4 camera->robot-base transform (.npy)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--o3d", action="store_true")
    ap.add_argument("--pybullet", action="store_true")
    ap.add_argument("--gui", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rig = StereoRig()
    if a.left and a.right:
        left, right = cv2.imread(a.left), cv2.imread(a.right)
        if left is None or right is None:
            sys.exit("could not read input images")
        if a.calib:
            from stereoedge.stereo.rectify import Rectifier
            rect = Rectifier(a.calib, (rig.width, rig.height), rig.num_disp)
            rig = rect.rig
            left, right = rect(left, right)
        else:
            print("[info] no --calib: assuming images are already rectified; using default intrinsics")
            left = cv2.resize(left, (rig.width, rig.height)); right = cv2.resize(right, (rig.width, rig.height))
        if a.handeye:
            rig.T_base_cam = np.load(a.handeye)
    else:
        from stereoedge.synth import make_scene
        scene = make_scene(rig, seed=a.seed)
        left, right = scene["left"], scene["right"]

    backend = make_backend(a.backend, rig, a.model)
    pipe = StereoGraspPipeline(rig, backend, GripperSpec())
    pipe.run(left, right)                       # warm-up (session init, caches)
    runs = sorted((pipe.run(left, right) for _ in range(5)), key=lambda r: r.timings_ms["total"])
    res = runs[len(runs) // 2]                  # report the median-latency run of 5
    T = rig.T_base_cam

    print(f"backend      : {backend.name}" + (f"  [{getattr(backend, 'label', '')}]" if hasattr(backend, 'label') else ""))
    print(f"objects found: {len(res.clusters)}   grasps: {len(res.grasps)}")
    for k, v in res.timings_ms.items():
        print(f"  {k:<16s}{v:8.1f} ms")
    print(f"  => {1000 / res.timings_ms['total']:.1f} FPS on this machine")
    if not res.best:
        sys.exit("no collision-free grasp found")
    best = res.best.to_dict(T)
    print("BEST GRASP (robot base frame):", json.dumps(best))

    with open(os.path.join(a.out, "grasps.json"), "w") as f:
        json.dump(dict(backend=backend.name, timings_ms=res.timings_ms,
                       best=best, all=[g.to_dict(T) for g in res.grasps]), f, indent=2)
    viz.stereo_pair_image(left, right, os.path.join(a.out, "stereo_pair.png"))
    viz.overview_figure(res, left, right, rig, backend.name, os.path.join(a.out, "overview.png"))
    viz.grasp_closeup(res, left, rig, os.path.join(a.out, "grasp_3d.png"))
    print(f"saved figures + grasps.json to {a.out}/")
    if a.gif:
        viz.grasp_animation(res, left, rig, os.path.join(a.out, "grasp_sequence.gif"))
        print("saved grasp_sequence.gif")
    if a.o3d:
        from stereoedge import viz_o3d
        viz_o3d.show(res, left, rig, save_ply=os.path.join(a.out, "cloud.ply"))
    if a.pybullet:
        from stereoedge.sim import pybullet_arm
        print("PyBullet:", pybullet_arm.run(res, rig, gui=a.gui))


if __name__ == "__main__":
    main()
