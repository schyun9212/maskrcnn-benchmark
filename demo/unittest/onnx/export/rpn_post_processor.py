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

# This unittest is not working on 1.1.1 version of onnxruntime
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
sample_image, t_width, t_height = transform_image(cfg, original_image)

ONNX_OPSET_VERSION = 10

VALIDATION_TYPE = "FILE" # FILE, IO

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

    def test_anchor_generator(self):
        from maskrcnn_benchmark.modeling.rpn.inference import make_rpn_postprocessor
        from maskrcnn_benchmark.modeling.rpn.anchor_generator import make_anchor_generator
        from maskrcnn_benchmark.modeling.box_coder import BoxCoder

        image_list = ImageList(sample_image.unsqueeze(0), [(sample_image.size(-2), sample_image.size(-1))])
        sample_features = coco_demo.model.backbone(image_list.tensors)

        anchor_generator = make_anchor_generator(cfg)
  
        objectness, rpn_box_regression = coco_demo.model.rpn.head(sample_features)
        anchors = anchor_generator(image_list, sample_features)
        
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
