import onnx
import onnx.shape_inference
from onnx import shape_inference

import argparse
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument("--model")

args = parser.parse_args()

onnx_model = onnx.load(args.model)
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))

