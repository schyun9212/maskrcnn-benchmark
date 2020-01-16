import torch
from maskrcnn_benchmark.structures.image_list import ImageList
from maskrcnn_benchmark.structures.bounding_box import BoxList

import onnxruntime as ort

from . import coco_demo, sample_image
from demo.onnx.utils import infer_shapes

class Backbone(torch.nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

    def forward(self, image):
        image_list = ImageList(image.unsqueeze(0), [(image.size(-2), image.size(-1))])

        result = coco_demo.model.backbone(image_list.tensors)
        return result

def test(path, opset_version):
    backbone = Backbone()
    backbone.eval()

    torch.onnx.export(backbone, sample_image, path,
                        verbose=False,
                        do_constant_folding=False,
                        opset_version=opset_version, input_names=["i_image"])

    infer_shapes(path, f"./demo/models/T_backbone_ops{opset_version}.shape.onnx")

    ort_session = ort.InferenceSession(path)
    input_feed = {ort_session.get_inputs()[0].name: sample_image.numpy()}
    result = ort_session.run(None, input_feed)

    return (result, backbone(sample_image))