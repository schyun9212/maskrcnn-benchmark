import torch
import io
import unittest

from maskrcnn_benchmark.structures.image_list import ImageList

from demo.unittest.onnx.export import ONNXExportTester, ONNX_OPSET_VERSION, VALIDATION_TYPE, cfg, coco_demo, sample_features, sample_proposals, t_width, t_height

class ROIBoxPredictorTester(ONNXExportTester):
    def test_feature_extractor(self):
        from maskrcnn_benchmark.structures.bounding_box import BoxList
        from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
        from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_predictors import make_roi_box_predictor
        
        class ROIBoxPredictor(torch.nn.Module):
            def __init__(self):
                super(ROIBoxPredictor, self).__init__()
                self.feature_extractor = make_roi_box_feature_extractor(cfg, 256)
                self.predictor = make_roi_box_predictor(cfg, self.feature_extractor.out_channels)
            
            def forward(self, features, proposals):
                bbox, objectness = proposals

                proposals = BoxList(bbox, (t_width, t_height), mode="xyxy")
                proposals.add_field("objectenss", objectness)

                x = self.feature_extractor(features, [proposals])
                class_logits, box_regression = self.predictor(x)

                return (class_logits, box_regression)

        roi_box_predictor = ROIBoxPredictor()
        roi_box_predictor.eval()
        
        inputs, outputs = self.run_model(roi_box_predictor, (sample_features, sample_proposals))

        if VALIDATION_TYPE == "IO":
            onnx_io = io.BytesIO()
        else:
            onnx_io = "./demo/onnx_test_models/feature_extractor.onnx"

        torch.onnx.export(roi_box_predictor, inputs, onnx_io,
                            verbose=False,
                            do_constant_folding=False,
                            input_names=["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "bbox", "objectness"],
                            opset_version=ONNX_OPSET_VERSION)
        
        self.ort_validate(onnx_io, inputs, outputs)
            
if __name__ == '__main__':
    unittest.main()
