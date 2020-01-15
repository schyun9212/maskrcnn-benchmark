# %%
import torch

from PIL import Image
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from maskrcnn_benchmark.structures.image_list import ImageList

from demo.utils import load_image, imshow, masking_image
from demo.transform import transform_image
import os

OVERWRITE_MODEL = False
TEST_IMAGE_PATH = "./sample.jpg"
MODEL_NAME = "e2e_mask_rcnn_R_50_FPN_1x_caffe2"
# MODEL_DEVICE = "cuda"
MODEL_DEVICE = "cpu"
MODEL_PATH = f"./{MODEL_NAME}_{MODEL_DEVICE}.onnx"
ONNX_OPSET_VERSION = 10

# %%
config_file = f"../configs/caffe2/{MODEL_NAME}.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", MODEL_DEVICE])
cfg.freeze()

coco_demo = COCODemo(
    cfg,
    confidence_threshold=0.7,
    min_image_size=800,
)
# %%
original_image = load_image(TEST_IMAGE_PATH)
image, t_width, t_height = transform_image(cfg, original_image)

height, width = original_image.shape[:-1]

# %%
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

import sys

def _register_custom_op():
    '''
    All arguments must be convert to a Tensor
    '''
    from torch.onnx.symbolic_helper import parse_args
    from torch.onnx.symbolic_opset9 import select, unsqueeze, squeeze, _cast_Long, reshape
    from torchvision.ops.roi_align import _RoIAlignFunction
    
    @parse_args('v', 'v', 'f')
    def symbolic_nms(g, boxes, scores, threshold):
        boxes = unsqueeze(g, boxes, 0)
        scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)
        max_output_per_class = g.op('Constant', value_t=torch.tensor([sys.maxsize], dtype=torch.long))
        threshold = g.op('Constant', value_t=torch.tensor([threshold], dtype=torch.float))
        nms_out = g.op('NonMaxSuppression', boxes, scores, max_output_per_class, threshold)
        return squeeze(g, select(g, nms_out, 1, g.op('Constant', value_t=torch.tensor([2], dtype=torch.long))), 1)

    @parse_args('v', 'v', 'f', 'i', 'i', 'i')
    def symbolic_roi_align_forward(g, grad, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
        batch_indices = _cast_Long(g, squeeze(g, select(g, rois, 1, g.op('Constant',
                                    value_t=torch.tensor([0], dtype=torch.long))), 1), False)
        rois = select(g, rois, 1, g.op('Constant', value_t=torch.tensor([1, 2, 3, 4], dtype=torch.long)))
        return g.op("RoiAlign", grad, rois, batch_indices, spatial_scale_f=spatial_scale, output_height_i=pooled_height, output_width_i=pooled_width, sampling_ratio_i=sampling_ratio)

    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic('maskrcnn_benchmark::nms', symbolic_nms, ONNX_OPSET_VERSION)
    register_custom_op_symbolic('maskrcnn_benchmark::roi_align_forward', symbolic_roi_align_forward, ONNX_OPSET_VERSION)

_register_custom_op()

# %%
if OVERWRITE_MODEL or not os.path.exists(MODEL_PATH):
    model = MaskRCNNModel()
    model.eval()
    torch.onnx.export(model, (image, ), MODEL_PATH,
                        verbose=False,
                        do_constant_folding=True,
                        opset_version=ONNX_OPSET_VERSION)
# %%
import onnx
import onnxruntime as ort

loaded_onnx_model = onnx.load(MODEL_PATH)
onnx.checker.check_model(loaded_onnx_model)
onnx.helper.printable_graph(loaded_onnx_model.graph)

# %%
ort_session = ort.InferenceSession(MODEL_PATH)
outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: image})

# %%
