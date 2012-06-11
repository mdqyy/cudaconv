#ifndef FCONV_H_
#define FCONV_H_

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>

void fconv(const std::vector<cv::gpu::DevMem2D_<float> > dev_features, 
           const std::vector<cv::gpu::DevMem2D_<float> > dev_filters, 
           std::vector<std::vector<cv::gpu::DevMem2D_<float> > > dev_responses);

#endif
