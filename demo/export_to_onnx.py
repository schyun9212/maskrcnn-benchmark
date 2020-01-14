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

from torch.onnx.symbolic_helper import parse_args
@parse_args('v', 'v', 'f')
def symbolic_nms(g, dets, scores, threshold):
    # Constant value must be converted to tensor
    threshold = g.op("Constant", value_t=torch.tensor(threshold, dtype=torch.float))
    return g.op("nms", dets, scores, threshold)

@parse_args('v', 'v', 'f', 'i', 'i', 'i')
def symbolic_roi_align_forward(g, grad, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    # Constant value must be converted to tensor
    spatial_scale = g.op("Constant", value_t=torch.tensor(spatial_scale, dtype=torch.float))
    pooled_height = g.op("Constant", value_t=torch.tensor(pooled_height, dtype=torch.int64))
    pooled_width = g.op("Constant", value_t=torch.tensor(pooled_width, dtype=torch.int64))
    sampling_ratio = g.op("Constant", value_t=torch.tensor(sampling_ratio, dtype=torch.int64))
    return g.op("roi_align_foward", grad, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio)

from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic('maskrcnn_benchmark::nms', symbolic_nms, 10)
register_custom_op_symbolic('maskrcnn_benchmark::roi_align_forward', symbolic_roi_align_forward, 10)

# %%
# requires_grad must be false for tracing
if OVERWRITE_MODEL or not os.path.exists(MODEL_PATH):
    model = MaskRCNNModel()
    model.eval()
    torch.onnx.export(model, (image, ), MODEL_PATH,
                        do_constant_folding=True,
                        opset_version=ONNX_OPSET_VERSION)
# %%
import onnx
import onnxruntime as ort

loaded_onnx_model = onnx.load('maskrcnn_cpu.onnx')
onnx.checker.check_model(loaded_onnx_model)
onnx.helper.printable_graph(loaded_onnx_model.graph)

# %%
ort_session = ort.InferenceSession(MODEL_PATH)
outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: image})
