import torch

from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo

from demo.utils import load_image
from demo.transform import transform_image

from demo.mask_rcnn import MaskRCNNModel

import unittest

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

class TorchScriptTraceTester(unittest.TestCase):
    def run_model(self, model, inputs):
        with torch.no_grad():
            if isinstance(inputs, torch.Tensor):
                inputs = (inputs,)
            outputs = model(*inputs)
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            
        return (inputs, outputs)

    def trace_model(self, model, inputs):
        model.eval()

        inputs, outputs = self.run_model(model, inputs)

        # export to scriptmodel
        traced_model = torch.jit.trace(model, inputs)

        ts_outputs = traced_model(*inputs)
        for i in range(0, len(outputs)):
            torch.testing.assert_allclose(outputs[i], ts_outputs[i], rtol=1e-02, atol=1e-04)

    def test_tracing(self):
        self.trace_model(MaskRCNNModel(cfg), sample_image)

if __name__ == '__main__':
    unittest.main()
