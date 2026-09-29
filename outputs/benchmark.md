Machine: Linux-6.18.44-fc-v37-x86_64-with-glibc2.39 (x86_64), Python 3.12.3, iters=30

| stage (median ms) | sgbm | onnx-cpu | onnx-qnn |
|---|---|---|---|
| stereo_matching | 8.6 | 24.4 | n/a (no QNN EP) |
| point_cloud | 1.8 | 1.8 | n/a (no QNN EP) |
| plane_fit | 14.5 | 14.1 | n/a (no QNN EP) |
| segmentation | 2.2 | 2.3 | n/a (no QNN EP) |
| normals | 18.6 | 13.8 | n/a (no QNN EP) |
| grasp_synthesis | 19.7 | 18.1 | n/a (no QNN EP) |
| total | 66.0 | 75.1 | n/a (no QNN EP) |
| FPS (1000/total) | 15.2 | 13.3 | n/a |
