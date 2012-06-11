// test the functionality provided by the cross correlation function
#include <omp.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "CCorr.hpp"
using namespace gpu;
/*
int main(int argc, char **argv) {
    // create a random matrix of noise
    int num_trials = 10000;
    cv::Mat image;
    cv::Mat imagef;
    cv::Mat cpart = cv::Mat::zeros(500, 500, CV_32FC1);
    std::vector<cv::Mat> cimage;
    image = cv::imread("/u/hbristow/willow/coffee.jpg");
    cv::cvtColor(image, image, CV_RGB2GRAY);
    image.convertTo(image, CV_32FC1, 1.0/255.0);
    cimage.push_back(image);
    cimage.push_back(cpart);
    cv::merge(cimage, imagef);
    //std::cout << "Image size " << imagef.size().width << ", " << imagef.size().height << std::endl;
    //std::cout << "Image type: " << imagef.type() << std::endl;
    // create a box filter to filter the image with
    cv::Mat filter = cv::Mat::zeros(500, 500, CV_32FC2);
    filter(cv::Rect(0,0,9,9)) = 1.0;
    //std::cout << "Filter size " << filter.size().width << ", " << filter.size().height << std::endl;
    //std::cout << "Filter type: " << filter.type() << std::endl;
    //std::vector<cv::Mat> filters;
    //  filters.push_back(filter);

    // find the cross correlation on the cpu
    std::vector<cv::Mat> responses;
    //openmp
    omp_set_num_threads(omp_get_num_procs());

    double t = (double)cv::getTickCount();
    for (int n = 0; n < num_trials; ++n) {
        cv::Mat filtercpu;
        cv::Mat imagecpu;
        cv::Mat response_cpu;
        cv::dft(imagef, imagecpu,cv::DFT_COMPLEX_OUTPUT);
        cv::dft(filter, filtercpu, cv::DFT_COMPLEX_OUTPUT);
        cv::mulSpectrums(imagecpu, filtercpu, imagecpu, cv::DFT_ROWS);
        cv::dft(imagecpu, response_cpu, cv::DFT_REAL_OUTPUT);
    }
    std::cout << "Single core cpu time: " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << std::endl;

    t = (double)cv::getTickCount();
#pragma omp parallel for
    for (int n = 0; n < num_trials; ++n) {
        cv::Mat filtercpu;
        cv::Mat imagecpu;
        cv::Mat response_cpu;
        cv::dft(imagef, imagecpu,cv::DFT_COMPLEX_OUTPUT);
        cv::dft(filter, filtercpu, cv::DFT_COMPLEX_OUTPUT);
        cv::mulSpectrums(imagecpu, filtercpu, imagecpu, cv::DFT_ROWS);
        cv::dft(imagecpu, response_cpu, cv::DFT_REAL_OUTPUT);
    }
    std::cout << "8 core cpu time: " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << std::endl;

    cv::Mat filtercpu = cv::Mat::ones(10, 10, CV_32FC1);
    t = (double)cv::getTickCount();
#pragma omp parallel for
    for (int n = 0; n < num_trials; ++n) {
        cv::Mat response_cpu;
        cv::filter2D(image, response_cpu, CV_32F, filtercpu);
    }
    std::cout << "fast 8 core cpu time: " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << std::endl;

    if (cv::gpu::getCudaEnabledDeviceCount()) {
        CCorr correlator_engine(filter, imagef.size());
        t = (double)cv::getTickCount();
        for (int n = 0; n < num_trials; ++n) {
            correlator_engine.correlate(imagef, responses);
        }
        std::cout << "naive gpu time: " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << std::endl;
        // display the original and the filtered image
        //cv::imshow("Original Image", image);
        //cv::imshow("Filter", filter);
        //cv::imshow("Filtered Image", responses[0]);
        //std::cout << responses[0].size().width << ", " << responses[0].size().height << std::endl;
        //cv::waitKey();

        t = (double)cv::getTickCount();
        cv::gpu::GpuMat imagegpu(image);
        cv::gpu::GpuMat filtergpu(cv::Mat::ones(10, 10, CV_32FC1));
        //std::cout << "Filter type: " << filtergpu.type() << " should be: " << CV_32F << std::endl;
        for (int n = 0; n < num_trials; ++n) {
            cv::gpu::GpuMat response_gpu;
            cv::gpu::convolve(imagegpu, filtergpu, response_gpu, true);
        }
        std::cout << "fast gpu time: " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << std::endl;


        // setup a GPU stream
        cv::gpu::Stream gpustream;
        cv::gpu::ConvolveBuf convbuffer;
        convbuffer.create(imagegpu.size(), filtergpu.size());
        t = (double)cv::getTickCount();
        for (int n = 0; n < num_trials; ++n) {
            cv::gpu::GpuMat response_gpu;
            cv::gpu::convolve(imagegpu, filtergpu, response_gpu, true, convbuffer, gpustream);
        }
        gpustream.waitForCompletion();
        std::cout << "fast streamed gpu time: " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << std::endl;

        
        t = (double)cv::getTickCount();
        for (int n = 0; n < num_trials; ++n) {
            cv::gpu::GpuMat response_gpu;
            cv::gpu::filter2D(imagegpu, response_gpu, CV_32F, filtercpu, cv::Point(-1,-1), cv::BORDER_DEFAULT, gpustream);
        }
        gpustream.waitForCompletion();
        std::cout << "fast streamed gpu time: " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << std::endl;




    }
}
*/
