import numpy as np
import torch

from demo.onnx.utils import infer_shapes
from demo.onnx.test import backbone, rpn
import demo.onnx.test

try:
    ONNX_OPSET_VERSION = 10

    MODULE = "backbone"
    BACKBONE_OPS10_PATH = "T_backbone_ops10.onnx"
    backbone_ops10, expected_val = backbone.test(BACKBONE_OPS10_PATH, ONNX_OPSET_VERSION)

    MODULE = "rpn"
    RPN_OPS10_PATH = "T_rpn_ops10.onnx"
    rpn_ops10, expected_val = rpn.test(RPN_OPS10_PATH, ONNX_OPSET_VERSION)

    ONNX_OPSET_VERSION = 11

    MODULE = "backbone"
    BACKBONE_OPS11_PATH = "T_backbone_ops11.onnx"
    backbone_ops11, _ = backbone.test(BACKBONE_OPS11_PATH, ONNX_OPSET_VERSION)

    MODULE = "rpn"
    RPN_OPS11_PATH = "T_rpn_ops11.onnx"
    rpn_ops11, _ = rpn.test(RPN_OPS11_PATH, ONNX_OPSET_VERSION)
except:
    print(f"failed to export {MODULE} with opset {ONNX_OPSET_VERSION}")