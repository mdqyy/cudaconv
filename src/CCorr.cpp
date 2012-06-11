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
#include <iostream>
#include "CCorr.hpp"
namespace gpu {

    void CCorr::correlate(const cv::Mat &image, std::vector<cv::Mat> &response) const {
        cv::gpu::GpuMat gpuimage(image);
        correlate(gpuimage, response);
    }

    void CCorr::correlate(const cv::gpu::GpuMat &gpuimage, std::vector<cv::Mat> &response) const {

        // check if there are any filters
        if (num_filters_ == 0) return;
        // get the size of the image
        cv::Size imsize = gpuimage.size();
        // fourier transform the image
        cv::gpu::GpuMat gpuimagef;
        cv::gpu::dft(gpuimage, gpuimagef, dft_size_, cv::DFT_INVERSE);
        //std::cout << "GPU image size: " << gpuimagef.size().height << ", " << gpuimagef.size().width << std::endl;
        //std::cout << "GPU Image type: " << gpuimagef.type() << std::endl;
        // multiply the spectrums
        //std::cout << filters_d_.size() << std::endl;

        for (int n = 0; n < filters_d_.size(); ++n) {
            assert(filters_d_[n].size() == gpuimagef.size() && filters_d_[n].type() == gpuimage.type());
            cv::gpu::GpuMat gpuresponsef;
            cv::gpu::GpuMat gpuresponse;
            cv::Mat responsen;
            cv::gpu::mulSpectrums(gpuimagef, filters_d_[n], gpuresponsef, 0);
            cv::gpu::dft(gpuresponsef, gpuresponse, gpuresponsef.size(), cv::DFT_REAL_OUTPUT);
            gpuresponse.download(responsen);
            response.push_back(responsen);
        }
    }

    void CCorr::copyFiltersHostToGPU(void) {
        //std::cout << "We copy the filters to the GPU" << std::endl;
        //std::cout << filters_h_.size() << std::endl;
        for (int n = 0; n < filters_h_.size(); ++n) {
            cv::gpu::GpuMat filter(filters_h_[n]);
            cv::gpu::dft(filter, filter, dft_size_, cv::DFT_INVERSE);
            filters_d_.push_back(cv::gpu::GpuMat(filter));
            //std::cout << "GPU Filter size: " << filter.size().height << ", " << filter.size().width << std::endl;
            //std::cout << "GPU Filter type: " << filter.type() << std::endl;
        }
    }
}
