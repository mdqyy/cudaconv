/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */
#include "fconv.hpp"
#include <cuda_runtime_api.h>
#include <cutil.h>
#include <cuda.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>


/* 
 * fconv
 * Is a specialised convolution routine for computing the responses to a 
 * set of features provided by the PartsBasedDetector
 */
 template <const unsigned int block_size>
__global__ void fconv_kernel(float **features, float **filters, float **responses, const unsigned int rwidth, const unsigned int rheight, const unsigned int feature_width, const unsigned int filter_width, const unsigned int filter_height) {

    // shared memory for the filter
    __shared__ float sfilter[2*block_size];
    // shared memory for the output of the multiplication
    __shared__ float sresponse[block_size];

    // get the thread indices into the filter
    const int idx = threadIdx.y*blockDim.x + threadIdx.x;

    // dereference this block's filter and feature
    const float *filter   = filters[blockIdx.x];
    const float *feature  = features[blockIdx.y];
    float *response = responses[blockIdx.y*gridDim.x + blockIdx.x]; 

    // initialise the response array
    sresponse[idx] = 0;

    // copy the filter into shared memory
    sfilter[idx]  = filter[idx];
    if (threadIdx.y+blockDim.y < filter_height) sfilter[idx+block_size] = filter[idx+block_size];
    __syncthreads();

    // loop over all heights
    for (int i = 0; i < rheight; ++i) {

        // loop over all widths
        for (int j = 0; j < rwidth; ++j) {
            
            sresponse[idx] =  feature[(i+threadIdx.y)*feature_width + j*filter_width + threadIdx.x] * sfilter[idx];
            if (threadIdx.y+blockDim.y < filter_height) sresponse[idx] +=  feature[(i+threadIdx.y+blockDim.y)*feature_width + j*filter_width + threadIdx.x] * sfilter[idx+block_size];
            __syncthreads();

            // reduce the response to a single value
            if (block_size >= 512)  {if (idx < 256)  {sresponse[idx] = sresponse[idx+256];  } __syncthreads(); }
            if (block_size >= 256)  {if (idx < 128)  {sresponse[idx] = sresponse[idx+128];  } __syncthreads(); }
            if (block_size >= 128)  {if (idx < 64)   {sresponse[idx] = sresponse[idx+64];   } __syncthreads(); }
            
            // unwrap last warp execution path so we no longer need to wait for
            // the other threads
            if (idx < 32) {
                if (block_size >= 64) sresponse[idx] = sresponse[idx+32];
                if (block_size >= 32) sresponse[idx] = sresponse[idx+16];
                if (block_size >= 16) sresponse[idx] = sresponse[idx+8];
                if (block_size >= 8)  sresponse[idx] = sresponse[idx+4];
                if (block_size >= 4)  sresponse[idx] = sresponse[idx+2];
                if (block_size >= 2)  sresponse[idx] = sresponse[idx+1];
            }
            // write the final result out to global memory
            if (idx == 0) response[j*rwidth+i] = sresponse[0];
        }
    }
    return;
}


void fconv(const std::vector<cv::gpu::DevMem2D_<float> > dev_features, 
           const std::vector<cv::gpu::DevMem2D_<float> > dev_filters, 
           std::vector<std::vector<cv::gpu::DevMem2D_<float> > > dev_responses) {

    // strip the data from the device pointers
    int nfeatures = dev_features.size();
    int nfilters  = dev_filters.size();

    float **features_d;
    float **features_h = new float *[nfeatures];
    CUDA_SAFE_CALL(cudaMalloc((void **)&features_d, sizeof(float *) * nfeatures));
    for (int i = 0; i < nfeatures; ++i) features_h[i] = dev_features[i].data;
    CUDA_SAFE_CALL(cudaMemcpy((void *)features_d, (void *)features_h, sizeof(float *) * nfeatures, cudaMemcpyHostToDevice));
    
    
    float **filters_d;
    float **filters_h  = new float *[nfilters];
    CUDA_SAFE_CALL(cudaMalloc((void **)&filters_d, sizeof(float *) * nfilters));
    for (int j = 0; j < nfilters; ++j) filters_h[j] = dev_filters[j].data;
    CUDA_SAFE_CALL(cudaMemcpy((void *)filters_d, (void *)filters_h, sizeof(float *) * nfilters, cudaMemcpyHostToDevice));

    
    assert(dev_responses.size() == nfeatures);
    float **responses_d;
    float **responses_h = new float *[nfeatures*nfilters];
    CUDA_SAFE_CALL(cudaMalloc((void **)&responses_d, sizeof(float *) * nfeatures*nfilters));
    for (int i = 0; i < nfeatures; ++i) {
        assert(dev_responses[i].size() == nfilters);
        for (int j = 0; j < nfilters; ++j) {
            responses_h[i*nfilters + j] = dev_responses[i][j].data;
        }
    }
    CUDA_SAFE_CALL(cudaMemcpy((void *)responses_d, (void *)responses_h, sizeof(float *) * nfeatures*nfilters, cudaMemcpyHostToDevice));
    
    // get the response height and width
    int rwidth  = dev_responses[0][0].cols;
    int rheight = dev_responses[0][0].rows; 

    // invoke the kernel
    dim3 grid(nfilters, nfeatures);
    //dim3 grid(1,1);
    dim3 threads(32*4, 4/2);
    //fconv_kernel<32, 32*4*4><<< grid, threads >>>(features, filters, responses, rwidth, rheight);
    fconv_kernel<32*4*4><<< grid, threads >>>(features_d, filters_d, responses_d, rwidth, rheight, 240*32, 32, 4);
    cudaThreadSynchronize();
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    // setdown
    /*
    free(features);
    free(filters);
    for (int i = 0; i < nfeatures; ++i) free(responses[i]);
    free(responses);
    */
}
