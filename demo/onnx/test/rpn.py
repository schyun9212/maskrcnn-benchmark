import torch
from maskrcnn_benchmark.structures.image_list import ImageList
from maskrcnn_benchmark.structures.bounding_box import BoxList

import onnxruntime as ort

from . import coco_demo, sample_image
from demo.onnx.utils import infer_shapes

class RPN(torch.nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, image, features):
        image_list = ImageList(image.unsqueeze(0), [(image.size(-2), image.size(-1))])
        result = coco_demo.model.rpn(image_list, features)[0][0]
        # rpn has extra field "objectness"
        result = (result.bbox,) + tuple(f for f in (result.get_field(field) for field in sorted(result.fields())) if isinstance(f, torch.Tensor))
        return result

def test(path, opset_version):
    rpn = RPN()
    rpn.eval()

    image_list = ImageList(sample_image.unsqueeze(0), [(sample_image.size(-2), sample_image.size(-1))])
    sample_features = coco_demo.model.backbone(image_list.tensors)

    torch.onnx.export(rpn, (sample_image, sample_features), path,
                        verbose=False,
                        do_constant_folding=False,
                        opset_version=opset_version, input_names=["i_image"] + [f"i_features_{i}" for i, _ in enumerate(sample_features)])

    infer_shapes(path, f"./demo/models/T_rpn_ops{opset_version}.shape.onnx")

    ort_session = ort.InferenceSession(path)
    input_feed = {
        "i_image": sample_image.numpy(),
        "i_features_0": sample_features[0].numpy(),
        "i_features_1": sample_features[1].numpy(),
        "i_features_2": sample_features[2].numpy(),
        "i_features_3": sample_features[3].numpy(),
        "i_features_4": sample_features[4].numpy(),
    }
    result = ort_session.run(None, input_feed)

    return (result, rpn(sample_image, sample_features))