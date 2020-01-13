// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/script.h>

#include "nms.h"
#include "ROIAlign.h"

static auto registry =
  torch::jit::RegisterOperators()
    .op("maskrcnn_benchmark::nms", &nms)
    .op("maskrcnn_benchmark::roi_align_forward(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) -> Tensor", &ROIAlign_forward);