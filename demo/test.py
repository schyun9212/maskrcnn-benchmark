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

#%%
config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
# cfg.freeze()

coco_demo = COCODemo(
    cfg,
    confidence_threshold=0.7,
    min_image_size=800,
)

# %%
original_image = load_image("./sample.jpg")
transformed_image, t_width, t_height = transform_image(cfg, original_image)

height, width = original_image.shape[:-1]

# Gradient must be deactivated
for p in coco_demo.model.parameters():
    p.requires_grad_(False)

# %%
def backbone_to_trace(image):
    '''
    Args:
        image (Tensor): Tensor image
        features (Tensor): Tensor feature resulted by backbone
    '''
    image_list = ImageList(image.unsqueeze(0), [(int(image.size(-2)), int(image.size(-1)))])
    result = coco_demo.model.backbone(image_list.tensors)
    return result

traced_backbone = torch.jit.trace(backbone_to_trace, transformed_image)
t_features = traced_backbone(transformed_image)

# %%
def rpn_to_trace(image, features):
    '''
    Args:
        original_image (Tensor): Tensor image
        features (Tensor): Tensor feature resulted by backbone
    '''
    image_list = ImageList(image.unsqueeze(0), [(int(image.size(-2)), int(image.size(-1)))])
    result = coco_demo.model.rpn(image_list, features)[0][0]
    # rpn has extra field "objectness"
    result = (result.bbox,) + tuple(f for f in (result.get_field(field) for field in sorted(result.fields())) if isinstance(f, torch.Tensor))
    return result

# %%
traced_rpn = torch.jit.trace(rpn_to_trace, (transformed_image, t_features))
t_proposals_bbox, t_proposals_objectness = traced_rpn(transformed_image, t_features)
t_proposals = BoxList(t_proposals_bbox, (t_width, t_height), mode="xyxy")
t_proposals.add_field("objectenss", t_proposals_objectness)

# %%
def roi_to_trace(features, proposals):
    bbox, objectness = proposals

    proposals = BoxList(bbox, (t_width, t_height), mode="xyxy")
    proposals.add_field("objectenss", objectness)

    _, result, _ = coco_demo.model.roi_heads(features, [proposals])

    result = (result[0].bbox,
            result[0].get_field("labels"),
            result[0].get_field("mask"),
            result[0].get_field("scores"))

    return result

# %%
traced_roi = torch.jit.trace(roi_to_trace, (t_features, (t_proposals_bbox, t_proposals_objectness)))
t_prediction_bbox, t_prediction_labels, t_prediction_mask, t_prediction_scores = traced_roi(t_features, (t_proposals_bbox, t_proposals_objectness))

# %%
t_prediction = BoxList(t_prediction_bbox, (t_width, t_height), mode="xyxy")
t_prediction.add_field("labels", t_prediction_labels),
t_prediction.add_field("mask", t_prediction_mask),
t_prediction.add_field("scores", t_prediction_scores)

# %%
t_prediction = t_prediction.resize((width, height))
traced_result = masking_image(coco_demo, original_image, t_prediction)
# %%
imshow(traced_result)

# %%
# Get original input to compare
image = coco_demo.transforms(original_image)
from maskrcnn_benchmark.structures.image_list import to_image_list
image_list = to_image_list(image, coco_demo.cfg.DATALOADER.SIZE_DIVISIBILITY)
image_list = image_list.to(coco_demo.device)

# Get original output to compare
features = coco_demo.model.backbone(image_list.tensors)
proposals = coco_demo.model.rpn(image_list, features)
_, predictions, _ = coco_demo.model.roi_heads(features, proposals[0])

# %%
prediction = predictions[0]
prediction = prediction.resize((width, height))
result = masking_image(coco_demo, original_image, prediction)
imshow(result)
