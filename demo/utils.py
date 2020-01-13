import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 20, 12

def load_image(path):
    image = Image.open(path).convert("RGB")
    # convert to BGR format
    image = np.array(image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")

def masking_image(coco_demo, original_image, prediction):
    if prediction.has_field("mask"):
        # if we have masks, paste the masks in the right position
        # in the image, as defined by the bounding boxes
        masks = prediction.get_field("mask")
        # always single image is passed at a time
        masks = coco_demo.masker([masks], [prediction])[0]
        prediction.add_field("mask", masks)

    top_prediction = coco_demo.select_top_predictions(prediction)

    result = original_image.copy()
    if coco_demo.show_mask_heatmaps:
        result = coco_demo.create_mask_montage(result, top_prediction)
    result = coco_demo.overlay_boxes(result, top_prediction)
    if coco_demo.cfg.MODEL.MASK_ON:
        result = coco_demo.overlay_mask(result, top_prediction)
    if coco_demo.cfg.MODEL.KEYPOINT_ON:
        result = coco_demo.overlay_keypoints(result, top_prediction)

    result = coco_demo.overlay_class_names(result, top_prediction)

    return result
