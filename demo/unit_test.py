# %%
import numpy as np

import requests
import torch

from PIL import Image
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from maskrcnn_benchmark.structures.image_list import ImageList
from maskrcnn_benchmark.structures.bounding_box import BoxList

from demo.utils import imshow, masking_image, load_image
from transform import transform_image

import onnx
import onnxruntime as ort

from demo.onnx.utils import infer_shapes

#%%
ONNX_OPSET_VERSION = 10

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
try:
    cfg.merge_from_file(config_file)
except:
    cfg.merge_from_file("./configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml")

cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.freeze()

coco_demo = COCODemo(
    cfg,
    confidence_threshold=0.7,
    min_image_size=800,
)

for param in coco_demo.model.parameters():
    param.requires_grad = False

# %%
try:
    original_image = load_image("./sample.jpg")
except:
    original_image = load_image("./demo/sample.jpg")
image, t_width, t_height = transform_image(cfg, original_image)

height, width = original_image.shape[:-1]

# Gradient must be deactivated
for p in coco_demo.model.parameters():
    p.requires_grad_(False)

# %%
BACKBONE_PATH = "backbone.onnx"

class Backbone(torch.nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

    def forward(self, image):
        image_list = ImageList(image.unsqueeze(0), [(int(image.size(-2)), int(image.size(-1)))])

        result = coco_demo.model.backbone(image_list.tensors)
        return result

backbone = Backbone()
backbone.eval()
expected_backbone_result = backbone(image)

torch.onnx.export(backbone, (image, ), BACKBONE_PATH,
                    verbose=False,
                    do_constant_folding=True,
                    opset_version=ONNX_OPSET_VERSION, input_names=["i_image"])

infer_shapes(BACKBONE_PATH, "backbone.shape.onnx")

ort_session = ort.InferenceSession(BACKBONE_PATH)
backbone_result = ort_session.run(None, {ort_session.get_inputs()[0].name: image.numpy()})
features = (torch.from_numpy(np.asarray(backbone_result[0])),)

# %%
RPN_PATH = "rpn.onnx"

class RPN(torch.nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, image, features):
        image_list = ImageList(image.unsqueeze(0), [(int(image.size(-2)), int(image.size(-1)))])
        result = coco_demo.model.rpn(image_list, features)[0][0]
        # rpn has extra field "objectness"
        result = (result.bbox,) + tuple(f for f in (result.get_field(field) for field in sorted(result.fields())) if isinstance(f, torch.Tensor))
        return result

rpn = RPN()
rpn.eval()
expected_rpn_result = rpn(image, features)

torch.onnx.export(rpn, (image, features), RPN_PATH,
                    verbose=False,
                    do_constant_folding=True,
                    opset_version=ONNX_OPSET_VERSION, input_names=["image", "features"])

infer_shapes(RPN_PATH, "rpn.shape.onnx")
# rpn_result = validate_model(RPN_PATH, {ort_session.get_inputs()[0].name: image.numpy()})

# ort_session = ort.InferenceSession("rpn.onnx")
# proposal_bboxs, proposal_objectnesses = ort_session.run(None, {
#     # "input_image": transformed_image.numpy(),
#     "input_features": features[0].numpy()
#     })
    
# %%
BACKBONE_RPN_PATH = "backbone+rpn.onnx"

class BackboneRPN(torch.nn.Module):
    def __init__(self):
        super(BackboneRPN, self).__init__()

    def forward(self, image):
        image_list = ImageList(image.unsqueeze(0), [(int(image.size(-2)), int(image.size(-1)))])

        features = coco_demo.model.backbone(image_list.tensors)
        result = coco_demo.model.rpn(image_list, features)[0][0]
        # rpn has extra field "objectness"
        result = (result.bbox,) + tuple(f for f in (result.get_field(field) for field in sorted(result.fields())) if isinstance(f, torch.Tensor))
        return result

backbone_rpn = BackboneRPN()
backbone_rpn.eval()
expected_backbone_rpn_result = backbone_rpn(image)

torch.onnx.export(backbone_rpn, (image, ), BACKBONE_RPN_PATH,
                    verbose=False,
                    do_constant_folding=True,
                    opset_version=ONNX_OPSET_VERSION, input_names=["image"])

infer_shapes(BACKBONE_PATH, "backbone+rpn.shape.onnx")
ort_session = ort.InferenceSession(BACKBONE_RPN_PATH)
backbone_rpn_result = ort_session.run(None, {ort_session.get_inputs()[0].name: image.numpy()})

# %%
ROI_PATH = "roi.onnx"

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
expected_roi_result = roi(expected_backbone_result, expected_rpn_result)

# %%
torch.onnx.export(roi, (expected_backbone_result, expected_backbone_rpn_result), ROI_PATH,
                    verbose=False,
                    do_constant_folding=True,
                    opset_version=ONNX_OPSET_VERSION, input_names=["image", "proposals"])

infer_shapes(ROI_PATH, "roi.shape.onnx")
# ort_session = ort.InferenceSession(BACKBONE_RPN_PATH)
# backbone_rpn_result = ort_session.run(None, {ort_session.get_inputs()[0].name: image.numpy()})
