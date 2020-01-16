import torch
import onnx
import onnx.shape_inference
import onnxruntime as ort

def infer_shapes(model_path, output_path="model.shape.onnx"):
    model = onnx.load(model_path)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)