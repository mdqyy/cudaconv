 #
 # Software License Agreement (BSD License)
 #
 #  Copyright (c) 2009, Willow Garage, Inc.
 #  All rights reserved.
 #
 #  Redistribution and use in source and binary forms, with or without
 #  modification, are permitted provided that the following conditions
 #  are met:
 #
 #   # Redistributions of source code must retain the above copyright
 #     notice, this list of conditions and the following disclaimer.
 #   # Redistributions in binary form must reproduce the above
 #     copyright notice, this list of conditions and the following
 #     disclaimer in the documentation and/or other materials provided
 #     with the distribution.
 #   # Neither the name of Willow Garage, Inc. nor the names of its
 #     contributors may be used to endorse or promote products derived
 #     from this software without specific prior written permission.
 #
 #  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 #  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 #  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 #  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 #  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 #  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 #  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 #  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 #  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 #  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 #  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 #  POSSIBILITY OF SUCH DAMAGE.
 #
 #
cmake_minimum_required(VERSION 2.8)
set (OPENCV_ROOT "/u/hbristow/dev")
set (CUDA_SDK_ROOT "/u/hbristow/dev/NVIDIA_GPU_Computing_SDK")
set (CMAKE_CXX_FLAGS "-Wall -fopenmp")
#set (CMAKE_VERBOSE_MAKEFILE ON)

include(FindCUDA)
#set (CUDA_BUILD_CUBIN true)

include_directories("${OPENCV_ROOT}/include")
include_directories("${CUDA_SDK_ROOT}/C/common/inc")
link_directories("${OPENCV_ROOT}/lib")
#link_libraries("-L/${CUDA_SDK_ROOT_DIR}/C/lib -lcutil")
#file(GLOB SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")
#file(GLOB HDR_FILES "${PROJECT_SOURCE_DIR}/src/*.hpp")

cuda_add_executable(conv src/fconv.cu src/fconv.cpp OPTIONS -arch=sm_21)
target_link_libraries(conv opencv_core opencv_imgproc opencv_highgui opencv_gpu)
