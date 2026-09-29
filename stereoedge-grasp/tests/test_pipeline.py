import os, sys, tempfile, warnings
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stereoedge.config import StereoRig
from stereoedge.synth import make_scene, OBJECTS
from stereoedge.inference.backend import make_backend, resolve_providers
from stereoedge.inference.onnx_model import build_stereo_onnx
from stereoedge.pipeline import StereoGraspPipeline

RIG = StereoRig()
MODEL = os.path.join(tempfile.gettempdir(), "se_test_model.onnx")


@pytest.fixture(scope="module")
def scene():
    return make_scene(RIG, seed=0)


def test_onnx_model_builds():
    build_stereo_onnx(MODEL, RIG.height, RIG.width, RIG.num_disp)
    assert os.path.getsize(MODEL) > 1000


@pytest.mark.parametrize("kind", ["sgbm", "onnx-cpu"])
def test_disparity_accuracy(scene, kind):
    d = make_backend(kind, RIG, MODEL).compute(scene["left"], scene["right"])
    v = np.isfinite(d)
    assert v.mean() > 0.75
    assert np.abs(d - scene["disp_gt"])[v].mean() < 0.5      # px


@pytest.mark.parametrize("kind", ["sgbm", "onnx-cpu"])
def test_grasps_are_valid_and_on_objects(scene, kind):
    res = StereoGraspPipeline(RIG, make_backend(kind, RIG, MODEL)).run(scene["left"], scene["right"])
    assert len(res.clusters) == 4
    assert abs(res.plane[1] - 0.55) < 0.015                    # table at ~0.55 m
    T = RIG.T_base_cam
    for c in res.clusters:
        lab = np.bincount(scene["labels"][c.mask]).argmax()
        g = [g for g in res.grasps if g.object_id == c.id]
        assert g, f"no grasp for {OBJECTS[lab][0]}"
        g = g[0]
        gb = g.transformed(T)
        assert 0.012 <= g.width <= 0.08
        assert gb.R[2, 2] < -0.97                              # approach points down
        assert abs(np.linalg.det(gb.R) - 1) < 1e-3             # proper rotation
        ctr = np.array(OBJECTS[lab][1])
        assert np.linalg.norm(g.center[:2] - ctr) < 0.02       # within 2 cm of true centre


def test_qnn_falls_back_to_cpu_when_missing():
    import onnxruntime as ort
    if "QNNExecutionProvider" in ort.get_available_providers():
        pytest.skip("QNN EP present")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        prov, label = resolve_providers("qnn")
    assert prov == ["CPUExecutionProvider"] and label == "CPU"


def test_rectifier_roundtrip(scene):
    from stereoedge.stereo.rectify import Rectifier
    K = np.array([[RIG.fx, 0, RIG.cx], [0, RIG.fy, RIG.cy], [0, 0, 1.0]])
    path = os.path.join(tempfile.gettempdir(), "se_calib.npz")
    np.savez(path, K1=K, D1=np.zeros(5), K2=K, D2=np.zeros(5), R=np.eye(3),
             T=np.array([-RIG.baseline, 0, 0]), size=np.array([RIG.width, RIG.height]))
    rect = Rectifier(path, (RIG.width, RIG.height), RIG.num_disp)
    assert abs(rect.rig.baseline - RIG.baseline) < 1e-4
    l, r = rect(scene["left"], scene["right"])
    d = make_backend("sgbm", rect.rig, MODEL).compute(l, r)
    v = np.isfinite(d)
    assert np.abs(d - scene["disp_gt"])[v].mean() < 0.6
