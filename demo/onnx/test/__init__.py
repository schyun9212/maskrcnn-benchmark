from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from demo.utils import load_image
from demo.transform import transform_image

import os

ONNX_OPSET_VERSION = 10

config_file = "./configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.freeze()

original_image = load_image("./demo/sample.jpg")
sample_image, t_width, t_height = transform_image(cfg, original_image)
height, width = original_image.shape[:-1]

coco_demo = COCODemo(
    cfg,
    confidence_threshold=0.7,
    min_image_size=800,
)

for param in coco_demo.model.parameters():
    param.requires_grad = False
