#!/usr/bin/env bash
set -e
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python tools/export_onnx.py
python -m pytest -q tests
python demo.py --backend onnx-cpu --gif
