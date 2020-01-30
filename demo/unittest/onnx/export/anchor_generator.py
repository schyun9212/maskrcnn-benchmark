import torch
import io
import unittest

from maskrcnn_benchmark.structures.image_list import ImageList

from demo.unittest.onnx.export import ONNXExportTester, ONNX_OPSET_VERSION, VALIDATION_TYPE, cfg, sample_image, sample_features

class AnchorGeneratorTester(ONNXExportTester):
    def test_anchor_generator(self):
        from maskrcnn_benchmark.modeling.rpn.anchor_generator import make_anchor_generator
        class AnchorGenerator(torch.nn.Module):
            def __init__(self):
                super(AnchorGenerator, self).__init__()
                self.anchor_generator = make_anchor_generator(cfg)

            def forward(self, image, features):
                """
                Arguments:
                    image (Tensor): images for which we want to compute the predictions
                    features (list[Tensor]): features computed from the images that are
                        used for computing the predictions. Each tensor in the list
                        correspond to different feature levels
                """
                image_list = ImageList(image.unsqueeze(0), [(image.size(-2), image.size(-1))])
                anchors = self.anchor_generator(image_list, features)
                
                result = [(x.bbox, x.get_field("visibility")) for x in anchors[0]]
                # anchor has "visibility" as extra field of BoxList
                return result

        anchor_generator = AnchorGenerator()
        anchor_generator.eval()
        
        inputs, outputs = self.run_model(anchor_generator, (sample_image, sample_features))

        if VALIDATION_TYPE == "IO":
            onnx_io = io.BytesIO()
        else:
            onnx_io = "./demo/onnx_test_models/rpn_anchor_generator.onnx"

        torch.onnx.export(anchor_generator, inputs, onnx_io,
                            verbose=False,
                            do_constant_folding=False,
                            input_names=["image", "feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
                            opset_version=ONNX_OPSET_VERSION)
        
        self.ort_validate(onnx_io, inputs, outputs)
        
if __name__ == '__main__':
    unittest.main()
