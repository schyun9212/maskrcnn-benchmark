# %%
import torch

from maskrcnn_benchmark.config import cfg

from demo.utils import load_image, imshow, masking_image
from demo.transform import transform_image
import os

from demo.mask_rcnn import MaskRCNNModel

OVERWRITE_MODEL = False
TEST_IMAGE_PATH = "./sample.jpg"
MODEL_NAME = "e2e_mask_rcnn_R_50_FPN_1x_caffe2"
MODEL_DEVICE = "cuda"
# MODEL_DEVICE = "cpu"
MODEL_REPOSITORY = f"./demo"
MODEL_PATH = f"{MODEL_REPOSITORY}/{MODEL_NAME}_{MODEL_DEVICE}.onnx"
ONNX_OPSET_VERSION = 10
config_file = f"../configs/caffe2/{MODEL_NAME}.yaml"

# %%
# original_image = load_image(TEST_IMAGE_PATH)
# image, t_width, t_height = transform_image(cfg, original_image)

# height, width = original_image.shape[:-1]

# %%
if OVERWRITE_MODEL or not os.path.exists(MODEL_PATH):
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.DEVICE", MODEL_DEVICE])
    cfg.freeze()
    model = MaskRCNNModel(cfg)
    model.eval()
    torch.onnx.export(model, (image, ), f"{MODEL_PATH}",
                        verbose=False,
                        do_constant_folding=True,
                        opset_version=ONNX_OPSET_VERSION)
# %%
import onnx
import onnx.shape_inference
from onnx import shape_inference

import onnxruntime as ort

onnx_model = onnx.load(MODEL_PATH)
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
onnx.checker.check_model(loaded_onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))

onnx.save(onnx_model, f"{MODEL_REPOSITORY}/{MODEL_NAME}_{MODEL_DEVICE}.shape.onnx")

# %%
# ort_session = ort.InferenceSession(MODEL_PATH)
# outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: image})

# %%
