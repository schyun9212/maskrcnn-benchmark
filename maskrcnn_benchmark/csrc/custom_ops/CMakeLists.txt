## IMPORTANT:
## This file is only of use when you want to build the custom ops and don't have
## python during the build.
## setup.py will build the custom ops all on its own without using this file

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(maskrcnn_benchmark_customops)


find_package(Torch REQUIRED NO_CMAKE_FIND_ROOT_PATH)

add_library(maskrcnn_benchmark_customops SHARED custom_ops.cpp ../cpu/nms_cpu.cpp ../cpu/ROIAlign_cpu.cpp)

target_compile_features(maskrcnn_benchmark_customops PUBLIC cxx_range_for)
#target_compile_options(maskrcnn_benchmark_customops PRIVATE "-std=c++11" "-D_GLIBCXX_USE_CXX11_ABI=1" "-std=gnu++11")
include_directories(BEFORE ../)
target_link_libraries(maskrcnn_benchmark_customops "${TORCH_LIBRARIES}")
target_compile_definitions(maskrcnn_benchmark_customops PRIVATE maskrcnn_benchmark_customops_EXPORTS)