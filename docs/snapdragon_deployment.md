# Snapdragon deployment guide

1. **Environment** – native ARM64 Python 3.11–3.13; `pip install -r requirements-snapdragon.txt`
   (`onnxruntime-qnn` provides `QNNExecutionProvider`). Check: `python -c "import onnxruntime as o; print(o.get_available_providers())"`.
2. **Run** – `python demo.py --backend onnx-qnn`. The backend passes `backend_path=QnnHtp.dll`,
   `htp_performance_mode=burst`, `enable_htp_fp16_precision=1` to the QNN EP (`inference/backend.py::resolve_providers`).
3. **Benchmark** – `python benchmarks/latency.py --iters 50`; the `onnx-qnn` column is measured on the NPU.
4. **Qualcomm AI Hub** – `pip install qai-hub`, configure your token, then `python tools/aihub_profile.py --device "Snapdragon X Elite CRD"`
   to compile and profile the model on a hosted Snapdragon device. Use the reported on-device latency for NPU claims.
5. **Swap in a trained model** – export a stereo/depth network to static-shape ONNX with inputs `left`,`right`
   and outputs `cost`,`disparity` (or edit `OnnxStereoBackend.compute` for a different output contract), quantise via AI Hub
   (INT8/W8A16) for best HTP throughput, and point `--model` at it.

Notes: ArgMin/Concat-heavy graphs may partially fall back to CPU inside the QNN EP; check the AI Hub profile for the op-level
placement before quoting NPU speed-ups. Open3D and PyBullet lack official Windows-ARM64 wheels; use the matplotlib renderer
and GIF on the device, and run the optional viewers on a separate x64/Linux/macOS machine from `outputs/grasps.json` / `cloud.ply`.
