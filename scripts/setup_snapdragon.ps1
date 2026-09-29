# Windows-on-ARM64 (HP / Snapdragon X) setup. Run from the project root in PowerShell.
# Needs a NATIVE ARM64 Python (python.org "Windows installer (ARM64)"), otherwise the QNN EP cannot load.
python -c "import platform; print('machine:', platform.machine())"   # expect ARM64
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-snapdragon.txt
python tools\export_onnx.py
python -c "import onnxruntime as o; print(o.get_available_providers())"   # expect QNNExecutionProvider
python demo.py --backend onnx-qnn --gif
python benchmarks\latency.py --iters 50
