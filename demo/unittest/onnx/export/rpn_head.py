import torch
import io
import unittest

from maskrcnn_benchmark.structures.image_list import ImageList

from demo.unittest.onnx.export import ONNXExportTester, ONNX_OPSET_VERSION, VALIDATION_TYPE, coco_demo, sample_features

class RPNHeadTester(ONNXExportTester):
    def test_rpn_head(self):
        class RPNHead(torch.nn.Module):
            def __init__(self):
                super(RPNHead, self).__init__()

            def forward(self, features):
                objectness, rpn_box_regression = coco_demo.model.rpn.head(features)
                return (objectness, rpn_box_regression)

        rpn_head = RPNHead()
        rpn_head.eval()
        
        inputs, outputs = self.run_model(rpn_head, (sample_features,))

        if VALIDATION_TYPE == "IO":
            onnx_io = io.BytesIO()
        else:
            onnx_io = "./demo/onnx_test_models/rpn_head.onnx"
        
        torch.onnx.export(rpn_head, inputs, onnx_io,
                            verbose=False,
                            do_constant_folding=False,
                            input_names=["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
                            opset_version=ONNX_OPSET_VERSION)
        
        self.ort_validate(onnx_io, inputs, outputs)
            
if __name__ == '__main__':
    unittest.main()
