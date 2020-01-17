import numpy as np
import torch
import io

from maskrcnn_benchmark.structures.image_list import ImageList
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.config import cfg

from demo.predictor import COCODemo
from demo.utils import load_image
from demo.transform import transform_image
from demo.onnx.utils import infer_shapes

import unittest
import onnxruntime

config_file = "./configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.freeze()

coco_demo = COCODemo(
    cfg,
    confidence_threshold=0.7,
    min_image_size=800,
)

for param in coco_demo.model.parameters():
    param.requires_grad = False

original_image = load_image("./demo/sample.jpg")
sample_image, _, _ = transform_image(cfg, original_image)

ONNX_OPSET_VERSION = 10

class ONNXExportTester(unittest.TestCase):
    def run_model(self, model, inputs):
        with torch.no_grad():
            if isinstance(inputs, torch.Tensor):
                inputs = (inputs,)
            outputs = model(*inputs)
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            
        return (inputs, outputs)

    def export_model(self, model, inputs):
        model.eval()

        inputs, outputs = self.run_model(model, inputs)

        onnx_io = io.BytesIO()
        # onnx_io = "test.onnx"

        # export to onnx
        torch.onnx.export(model, inputs, onnx_io, do_constant_folding=True, opset_version=ONNX_OPSET_VERSION)
        self.ort_validate(onnx_io, inputs, outputs)

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
            torch.testing.assert_allclose(outputs[i], ort_outs[i], rtol=1e-02, atol=1e-04)

    def test_backbone(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super(Backbone, self).__init__()

            def forward(self, image):
                image_list = ImageList(image.unsqueeze(0), [(image.size(-2), image.size(-1))])

                result = coco_demo.model.backbone(image_list.tensors)
                return result
        
        self.export_model(Backbone(), sample_image)

    def test_rpn(self):
        class RPN(torch.nn.Module):
            def __init__(self):
                super(RPN, self).__init__()

            def forward(self, image, features):
                image_list = ImageList(image.unsqueeze(0), [(image.size(-2), image.size(-1))])
                result = coco_demo.model.rpn(image_list, features)[0][0]
                # rpn has extra field "objectness"
                result = (result.bbox,) + tuple(f for f in (result.get_field(field) for field in sorted(result.fields())) if isinstance(f, torch.Tensor))
                return result

        image_list = ImageList(sample_image.unsqueeze(0), [(sample_image.size(-2), sample_image.size(-1))])
        sample_features = coco_demo.model.backbone(image_list.tensors)

        rpn = RPN()
        rpn.eval()
        
        inputs, outputs = self.run_model(rpn, (sample_image, sample_features))

        onnx_io = io.BytesIO()
        torch.onnx.export(rpn, inputs, onnx_io,
                            verbose=False,
                            do_constant_folding=False,
                            opset_version=ONNX_OPSET_VERSION)
        
        self.ort_validate(onnx_io, inputs, outputs)
        
        '''
        # The code that runs in this scope and that of export_model is the same,
        # but recognizes input differently

        # To reproduce
        # self.export_model(RPN(), (sample_image, sample_features))
        '''

if __name__ == '__main__':
    unittest.main()
