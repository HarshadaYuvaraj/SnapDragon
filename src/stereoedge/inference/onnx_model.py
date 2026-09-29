"""Builds a static-shape ONNX stereo-matching network (cost volume + aggregation).

Every op (Pad, Slice, Sub, Abs, Concat, AveragePool, ArgMin) is a standard tensor op, so the
same graph runs on onnxruntime-CPU and is a candidate for the Hexagon NPU via the QNN
Execution Provider / Qualcomm AI Hub. Weights are fixed (classical SAD cost volume), and the
graph is the drop-in slot for a *trained* stereo network (e.g. a HITNet / RAFT-Stereo variant
exported at the same input/output contract: left, right -> cost / disparity).
"""
import numpy as np
import onnx
from onnx import helper, TensorProto as TP, numpy_helper


def build_stereo_onnx(path, height=240, width=320, num_disp=48, window=7):
    D, k = num_disp, window
    nodes, inits = [], []

    def const(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    # Left-pad the right image so shifted slices keep static shape.
    pads = const("pads", np.array([0, 0, 0, D - 1, 0, 0, 0, 0], np.int64))
    padval = const("padval", np.array(1.0, np.float32))
    nodes.append(helper.make_node("Pad", ["right", pads, padval], ["right_pad"], mode="constant"))
    axes3 = const("axes3", np.array([3], np.int64))

    diffs = []
    for d in range(D):
        s = const(f"s{d}", np.array([D - 1 - d], np.int64))
        e = const(f"e{d}", np.array([D - 1 - d + width], np.int64))
        nodes += [
            helper.make_node("Slice", ["right_pad", s, e, axes3], [f"rs{d}"]),
            helper.make_node("Sub", ["left", f"rs{d}"], [f"df{d}"]),
            helper.make_node("Abs", [f"df{d}"], [f"ad{d}"]),
        ]
        diffs.append(f"ad{d}")
    nodes.append(helper.make_node("Concat", diffs, ["raw_cost"], axis=1))
    nodes.append(helper.make_node("AveragePool", ["raw_cost"], ["cost"],
                                  kernel_shape=[k, k], pads=[k // 2] * 4, strides=[1, 1],
                                  count_include_pad=0))
    nodes.append(helper.make_node("ArgMin", ["cost"], ["disp_i"], axis=1, keepdims=0))
    nodes.append(helper.make_node("Cast", ["disp_i"], ["disparity"], to=TP.FLOAT))

    g = helper.make_graph(
        nodes, "stereoedge_costvolume",
        [helper.make_tensor_value_info("left", TP.FLOAT, [1, 1, height, width]),
         helper.make_tensor_value_info("right", TP.FLOAT, [1, 1, height, width])],
        [helper.make_tensor_value_info("cost", TP.FLOAT, [1, D, height, width]),
         helper.make_tensor_value_info("disparity", TP.FLOAT, [1, height, width])],
        initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return path
