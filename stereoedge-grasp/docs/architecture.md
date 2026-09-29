# Architecture

```
 left cam ─┐                                         ┌── plane RANSAC ── object segmentation
           ├─ rectify ─ stereo matching (NPU) ─ disparity ─ point cloud ─┤
 right cam ┘   (CPU)     ONNX / QNN HTP            (CPU)                 └── PCA normals
                                                                                │
        6-DoF grasp (camera + base frame) ◄─ antipodal search ◄─ contour normals ◄┘
        + score + width                        + finger/table collision check (CPU)
```

## Frames
Camera: x right, y down, z forward. Robot base: table plane z = 0, z up. `StereoRig.T_base_cam` maps camera → base
(default: camera 0.55 m above the table looking straight down, 0.50 m in front of the robot).
Grasp rotation columns: closing axis, lateral axis, approach vector (approach = −z_base for top-down grasps).

## Backend contract
`StereoBackend.compute(left_bgr, right_bgr) -> float32 disparity (H,W)`, NaN where invalid. ONNX models take `left`,`right`
`[1,1,H,W]` float32 in [0,1] and output `cost [1,D,H,W]`, `disparity [1,H,W]`; sub-pixel refinement and validity gating are done
on the host. Static shapes are used because QNN/HTP graphs must be static.

## Latency
`pipeline.py` times every stage; `benchmarks/latency.py` reports median/p95 over N runs. Only the stereo stage is a
neural workload today; all other stages are CPU NumPy/SciPy, which is why end-to-end FPS is limited by them, not by the NPU.
