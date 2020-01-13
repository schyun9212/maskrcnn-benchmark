# %%
import torch

from PIL import Image
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from maskrcnn_benchmark.structures.image_list import ImageList

from demo.utils import load_image, imshow, masking_image
from demo.transform import transform_image
import os

OVERWRITE_MODEL = True
TEST_IMAGE_PATH = "./sample.jpg"
# MODEL_DEVICE = "cuda"
MODEL_DEVICE = "cpu"
MODEL_PATH = f"./script_model_{MODEL_DEVICE}.pt"

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", MODEL_DEVICE])
cfg.freeze()

coco_demo = COCODemo(
    cfg,
    confidence_threshold=0.7,
    min_image_size=800,
)

class MaskRCNNModel(torch.nn.Module):
    def __init__(self):
        super(MaskRCNNModel, self).__init__()
        for param in coco_demo.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        image_list = ImageList(image.unsqueeze(0), [(int(image.size(-2)), int(image.size(-1)))])

        result, = coco_demo.model(image_list)

        result = (result.bbox,
                result.get_field("labels"),
                result.get_field("mask"),
                result.get_field("scores"))
        return result

original_image = load_image(TEST_IMAGE_PATH)
image, t_width, t_height = transform_image(cfg, original_image)

height, width = original_image.shape[:-1]

# requires_grad must be false for tracing
if OVERWRITE_MODEL or not os.path.exists(MODEL_PATH):
    model = MaskRCNNModel()
    model.eval()
    traced_model = torch.jit.trace(model, (image,))
    traced_model.save(MODEL_PATH)

# %%
loaded_model = torch.jit.load(MODEL_PATH)
boxes, labels, mask, scores = loaded_model(image)

from maskrcnn_benchmark.structures.bounding_box import BoxList
prediction = BoxList(boxes, (t_width, t_height), mode="xyxy")
prediction.add_field("labels", labels),
prediction.add_field("mask", mask),
prediction.add_field("scores", scores)
prediction = prediction.resize((width, height))

result = masking_image(coco_demo, original_image, prediction)
imshow(result)

# %%