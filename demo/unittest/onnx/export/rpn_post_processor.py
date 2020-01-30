import torch
import io
import unittest

from maskrcnn_benchmark.structures.image_list import ImageList

from demo.unittest.onnx.export import ONNXExportTester, ONNX_OPSET_VERSION, VALIDATION_TYPE, cfg, coco_demo, sample_image, sample_image_list, sample_features

class RPNPostProcessorTester(ONNXExportTester):
    def test_anchor_generator(self):
        from maskrcnn_benchmark.modeling.rpn.inference import make_rpn_postprocessor
        from maskrcnn_benchmark.modeling.rpn.anchor_generator import make_anchor_generator
        from maskrcnn_benchmark.modeling.box_coder import BoxCoder

        anchor_generator = make_anchor_generator(cfg)
        objectness, rpn_box_regression = coco_demo.model.rpn.head(sample_features)
        anchors = anchor_generator(sample_image_list, sample_features)
        
        class RPNPostProcessor(torch.nn.Module):
            def __init__(self):
                super(RPNPostProcessor, self).__init__()
                rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

                # self.anchor_generator = make_anchor_generator(cfg)
                self.box_selector = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

            def forward(self, image, features):
                """
                Arguments:
                    image (Tensor): images for which we want to compute the predictions
                    features (list[Tensor]): features computed from the images that are
                        used for computing the predictions. Each tensor in the list
                        correspond to different feature levels
                """
                # image_list = ImageList(image.unsqueeze(0), [(image.size(-2), image.size(-1))])

                # objectness, rpn_box_regression = coco_demo.model.rpn.head(features)
                # anchors = self.anchor_generator(image_list, features)
                boxes = self.box_selector(anchors, objectness, rpn_box_regression)[0]

                return (boxes.bbox, boxes.get_field("objectness"))

        rpn_post_processor = RPNPostProcessor()
        rpn_post_processor.eval()
        
        inputs, outputs = self.run_model(rpn_post_processor, (sample_image, sample_features))

        if VALIDATION_TYPE == "IO":
            onnx_io = io.BytesIO()
        else:
            onnx_io = "./demo/onnx_test_models/rpn_post_processor.onnx"

        torch.onnx.export(rpn_post_processor, inputs, onnx_io,
                            verbose=False,
                            do_constant_folding=False,
                            input_names=["image", "feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
                            opset_version=ONNX_OPSET_VERSION)
        
        self.ort_validate(onnx_io, inputs, outputs)
            
if __name__ == '__main__':
    unittest.main()
