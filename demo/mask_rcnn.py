import torch

from predictor import COCODemo
from maskrcnn_benchmark.structures.image_list import ImageList

class MaskRCNNModel(torch.nn.Module):
    def __init__(self, cfg):
        super(MaskRCNNModel, self).__init__()

        self.coco_demo = COCODemo(
            cfg,
            confidence_threshold=0.7,
            min_image_size=800,
        )

        for param in self.coco_demo.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        image_list = ImageList(image.unsqueeze(0), [(int(image.size(-2)), int(image.size(-1)))])

        result, = self.coco_demo.model(image_list)

        result = (result.bbox,
                result.get_field("labels"),
                result.get_field("mask"),
                result.get_field("scores"))
        return result