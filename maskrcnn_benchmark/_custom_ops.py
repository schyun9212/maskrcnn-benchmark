import os

import torch

torch.ops.load_library(os.path.join(os.path.dirname(__file__), 'lib', 'libmaskrcnn_benchmark_customops.so'))

def _register_custom_op():
    '''
    All arguments must be convert to a Tensor
    '''
    import sys

    from torch.onnx.symbolic_helper import parse_args
    from torch.onnx.symbolic_opset9 import select, unsqueeze, squeeze, _cast_Long, reshape
    from torchvision.ops.roi_align import _RoIAlignFunction

    ONNX_OPSET_VERSION = 10
    
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