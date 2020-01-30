import torch
import io
import unittest

from maskrcnn_benchmark.structures.image_list import ImageList

from demo.unittest.onnx.export import ONNXExportTester, ONNX_OPSET_VERSION, VALIDATION_TYPE, cfg, coco_demo, sample_image, sample_features

class RPNTester(ONNXExportTester):
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

        rpn = RPN()
        rpn.eval()
        
        inputs, outputs = self.run_model(rpn, (sample_image, sample_features))

        if VALIDATION_TYPE == "IO":
            onnx_io = io.BytesIO()
        else:
            onnx_io = "./demo/onnx_test_models/rpn.onnx"

        torch.onnx.export(rpn, inputs, onnx_io,
                            verbose=False,
                            do_constant_folding=False,
                            input_names=["image", "feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
                            opset_version=ONNX_OPSET_VERSION)
        
        self.ort_validate(onnx_io, inputs, outputs)
            
if __name__ == '__main__':
    unittest.main()
