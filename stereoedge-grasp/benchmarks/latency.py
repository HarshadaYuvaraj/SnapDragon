#!/usr/bin/env python3
"""Per-stage latency benchmark. Run on the target machine; writes outputs/benchmark.{json,md}.

  python benchmarks/latency.py --iters 30
On a Snapdragon X PC with onnxruntime-qnn installed, the onnx-qnn row is measured on the NPU;
elsewhere it is reported as unavailable (never estimated).
"""
import argparse, json, os, platform, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from stereoedge.config import StereoRig
from stereoedge.synth import make_scene
from stereoedge.inference.backend import make_backend
from stereoedge.pipeline import StereoGraspPipeline


def qnn_available():
    try:
        import onnxruntime as ort
        return "QNNExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--model", default="models/stereo_costvolume.onnx")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rig = StereoRig(); s = make_scene(rig)
    rows, info = {}, dict(machine=platform.machine(), system=platform.platform(),
                          python=platform.python_version(), qnn_available=qnn_available())
    for kind in ["sgbm", "onnx-cpu", "onnx-qnn"]:
        if kind == "onnx-qnn" and not qnn_available():
            rows[kind] = None
            continue
        pipe = StereoGraspPipeline(rig, make_backend(kind, rig, a.model))
        for _ in range(3):
            pipe.run(s["left"], s["right"])
        runs = [pipe.run(s["left"], s["right"]).timings_ms for _ in range(a.iters)]
        rows[kind] = {k: dict(median=float(np.median([r[k] for r in runs])),
                              p95=float(np.percentile([r[k] for r in runs], 95))) for k in runs[0]}
    json.dump(dict(info=info, results=rows), open(os.path.join(a.out, "benchmark.json"), "w"), indent=2)

    stages = ["stereo_matching", "point_cloud", "plane_fit", "segmentation", "normals",
              "grasp_synthesis", "total"]
    md = [f"Machine: {info['system']} ({info['machine']}), Python {info['python']}, iters={a.iters}\n",
          "| stage (median ms) | " + " | ".join(rows) + " |", "|---|" + "---|" * len(rows)]
    for st in stages:
        md.append(f"| {st} | " + " | ".join("n/a (no QNN EP)" if rows[k] is None else f"{rows[k][st]['median']:.1f}"
                                            for k in rows) + " |")
    md.append("| FPS (1000/total) | " + " | ".join("n/a" if rows[k] is None else f"{1000 / rows[k]['total']['median']:.1f}"
                                                   for k in rows) + " |")
    open(os.path.join(a.out, "benchmark.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
