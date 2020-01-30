import torch
import io
import unittest

from maskrcnn_benchmark.structures.image_list import ImageList

from demo.unittest.onnx.export import ONNXExportTester, ONNX_OPSET_VERSION, VALIDATION_TYPE, coco_demo, sample_image

class BackboneTester(ONNXExportTester):
    def test_backbone(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super(Backbone, self).__init__()

            def forward(self, image):
                image_list = ImageList(image.unsqueeze(0), [(image.size(-2), image.size(-1))])

                result = coco_demo.model.backbone(image_list.tensors)
                return result

        backbone = Backbone()
        backbone.eval()
        inputs, outputs = self.run_model(backbone, (sample_image,))

        if VALIDATION_TYPE == "IO":
            onnx_io = io.BytesIO()
        else:
            onnx_io = "./demo/onnx_test_models/backbone.onnx"

        torch.onnx.export(backbone, inputs, onnx_io,
                            verbose=False,
                            do_constant_folding=False,
                            input_names=["image"],
                            opset_version=ONNX_OPSET_VERSION)
        
        self.ort_validate(onnx_io, inputs, outputs)
            
if __name__ == '__main__':
    unittest.main()
