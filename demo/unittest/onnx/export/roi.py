import torch
import io
import unittest

from maskrcnn_benchmark.structures.image_list import ImageList

from demo.unittest.onnx.export import ONNXExportTester, ONNX_OPSET_VERSION, VALIDATION_TYPE, cfg, coco_demo, sample_features, sample_proposals, t_width, t_height

class ROITester(ONNXExportTester):
    def test_roi(self):
        from maskrcnn_benchmark.structures.bounding_box import BoxList

        class ROI(torch.nn.Module):
            def __init__(self):
                super(ROI, self).__init__()

            def forward(self, features, proposals):
                bbox, objectness = proposals

                proposals = BoxList(bbox, (t_width, t_height), mode="xyxy")
                proposals.add_field("objectenss", objectness)

                _, result, _ = coco_demo.model.roi_heads(features, [proposals])

                result = (result[0].bbox,
                        result[0].get_field("labels"),
                        result[0].get_field("mask"),
                        result[0].get_field("scores"))

                return result

        roi = ROI()
        roi.eval()
        
        inputs, outputs = self.run_model(roi, (sample_features, sample_proposals))

        if VALIDATION_TYPE == "IO":
            onnx_io = io.BytesIO()
        else:
            onnx_io = "./demo/onnx_test_models/roi.onnx"

        torch.onnx.export(roi, inputs, onnx_io,
                            verbose=False,
                            do_constant_folding=False,
                            input_names=["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "bbox", "objectness"],
                            opset_version=ONNX_OPSET_VERSION)
        
        self.ort_validate(onnx_io, inputs, outputs)
            
if __name__ == '__main__':
    unittest.main()
