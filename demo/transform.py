from torchvision import transforms as T
from torchvision.transforms import functional as F

import torch

from predictor import Resize

def build_transform(cfg):
    """
    Creates a basic transformation that was used to train the models
    """

    # we are loading images with OpenCV, so we don't need to convert them
    # to BGR, they are already! So all we need to do is to normalize
    # by 255 if we want to convert to BGR255 format, or flip the channels
    # if we want it to be in RGB in [0-1] range.
    if cfg.INPUT.TO_BGR255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    transform = T.Compose(
        [
            T.ToPILImage(),
            Resize(min_size, max_size),
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )
    return transform

def transform_image(cfg, original_image):
    '''
    Args:
        cfg (CfgNode): configure of model
        original_image (Tensor): Tensor image

    Returns:
        Tensor: Transformed Tensor image
        width: Transformed Tensor width not consider SIZE_DIVISIBILITY
        height: Transformed Tensor height not consider SIZE_DIVISIBILITY
    '''
    transforms = build_transform(cfg)
    image = transforms(original_image)
    from maskrcnn_benchmark.structures.image_list import to_image_list
    image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)
    image_list = image_list.to(torch.device(cfg.MODEL.DEVICE))
    return (image_list.tensors[0], image.size(-1), image.size(-2))
