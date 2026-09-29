#!/usr/bin/env python3
"""Compile + profile the ONNX stereo model on a real Snapdragon device via Qualcomm AI Hub.

Prerequisites (not executed in the development sandbox - requires your API token + network):
    pip install qai-hub
    qai-hub configure --api_token <YOUR_TOKEN>      # https://aihub.qualcomm.com
    python tools/aihub_profile.py --device "Snapdragon X Elite CRD"
Use `python -c "import qai_hub as hub; [print(d.name) for d in hub.get_devices()]"` to list devices.
The measured on-device latency printed here is what should be quoted for NPU performance.
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/stereo_costvolume.onnx")
ap.add_argument("--device", default="Snapdragon X Elite CRD")
ap.add_argument("--height", type=int, default=240); ap.add_argument("--width", type=int, default=320)
a = ap.parse_args()

import qai_hub as hub

device = hub.Device(a.device)
spec = {"left": ((1, 1, a.height, a.width), "float32"), "right": ((1, 1, a.height, a.width), "float32")}
model = hub.upload_model(a.model)
compile_job = hub.submit_compile_job(model=model, device=device, input_specs=spec,
                                     options="--target_runtime onnx")
compile_job.wait()
profile_job = hub.submit_profile_job(model=compile_job.get_target_model(), device=device)
profile_job.wait()
prof = profile_job.download_profile()
summary = prof.get("execution_summary", {})
print("compile job :", compile_job.url)
print("profile job :", profile_job.url)
print("inference (us):", summary.get("estimated_inference_time"))
print("peak memory   :", summary.get("inference_memory_peak_range"))
