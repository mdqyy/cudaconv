/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
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
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/imgproc/imgproc.hpp>
namespace gpu {

    class Conv {
    private:
        // the filters on the host
        std::vector<cv::Mat> filters_h_;
        int num_filters_;
        // the filters on the GPU
        std::vector<cv::gpu::GpuMat> filters_d_;
        // copy operations between Host (CPU) and GPU
        void copyFiltersHostToGPU(void);
    public:
        // constructors
        Conv() {}
        Conv(const std::vector<cv::Mat> &filters) :
                filters_h_(filters), num_filters_(filters.size()) {
            copyFiltersHostToGPU();
        }
        Conv(const cv::Mat &filter) :
                filters_h_(std::vector<cv::Mat>(filter)), num_filters_(1) {
            copyFiltersHostToGPU();
        }
        // destructor
        ~Conv() {}

        // convolution of an image with a set of filters
        void convolve(const cv::Mat &image, std::vector<cv::Mat> &responses);
        void convolve(const cv::gpu::GpuMat &image, std::vector<cv::Mat> &responses);
        
        // helper functions
        void setFilters(std::vector<cv::Mat> &filters) {
            filters_h_ = filters;
            num_filters_ = filters.size();
            copyFiltersHostToGPU();
        }
        void setFilters(cv::Mat &filter) {
            filters_h_ = std::vector<cv::Mat>(filter);
            num_filters_ = 1;
            copyFiltersHostToGPU();
        }
        int getNumFilters() {
            return num_filters_;
        }
        std::vector<cv::Mat> getFilters() {
            return filters_h_;
        }
    };
}


