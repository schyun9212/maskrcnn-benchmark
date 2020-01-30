import numpy as np
import torch
import os

from maskrcnn_benchmark.structures.image_list import ImageList
from maskrcnn_benchmark.config import cfg

from demo.predictor import COCODemo
from demo.utils import load_image
from demo.transform import transform_image
from demo.onnx.utils import infer_shapes

import unittest

# This unittest is not working on 1.1.1 version of onnxruntime
import onnxruntime

CONFIG_FILE = "./configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
SAMPLE_IMAGE_PATH = "./demo/sample.jpg"
ONNX_OPSET_VERSION = 10
VALIDATION_TYPE = "FILE" # FILE, IO
ONNX_TEST_MODEL_PATH = "./demo/onnx_test_models"

if VALIDATION_TYPE == "FILE" and not os.path.exists(ONNX_TEST_MODEL_PATH):
    os.makedirs(ONNX_TEST_MODEL_PATH)

cfg.merge_from_file(CONFIG_FILE)
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.freeze()

coco_demo = COCODemo(
    cfg,
    confidence_threshold=0.7,
    min_image_size=800,
)

for param in coco_demo.model.parameters():
    param.requires_grad = False

original_image = load_image(SAMPLE_IMAGE_PATH)
sample_image, t_width, t_height = transform_image(cfg, original_image)

sample_image_list = ImageList(sample_image.unsqueeze(0), [(sample_image.size(-2), sample_image.size(-1))])
sample_features = coco_demo.model.backbone(sample_image_list.tensors)

class ONNXExportTester(unittest.TestCase):
    def run_model(self, model, inputs):
        with torch.no_grad():
            if isinstance(inputs, torch.Tensor):
                inputs = (inputs,)
            outputs = model(*inputs)
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            
        return (inputs, outputs)

    def ort_validate(self, onnx_io, inputs, outputs):
        inputs, _ = torch.jit._flatten(inputs)
        outputs, _ = torch.jit._flatten(outputs)

        def to_numpy(tensor):
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.cpu().numpy()

        inputs = list(map(to_numpy, inputs))
        outputs = list(map(to_numpy, outputs))

        if isinstance(onnx_io, str):
            ort_session = onnxruntime.InferenceSession(onnx_io)
        else:
            ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())

        # compute onnxruntime output prediction
        ort_inputs = dict((ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
        ort_outs = ort_session.run(None, ort_inputs)

        for i in range(0, len(outputs)):
            torch.testing.assert_allclose(outputs[i].astype(np.float32), ort_outs[i].astype(np.float32), rtol=1e-02, atol=1e-04)
