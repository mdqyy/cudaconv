// test the functionality provided by the gpu fconv method
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "fconv.hpp"
using namespace std;
using namespace cv;
using namespace cv::gpu;

int main(int argc, char **argv) {

    int nfilters = 500;
    int nfeatures = 8;

    vector<DevMem2D_<float> > filters;
    vector<DevMem2D_<float> > features;
    vector<GpuMat > filters_gm;
    vector<GpuMat > features_gm;
    vector<vector<DevMem2D_<float> > > responses;

    for (int i = 0; i < nfeatures; ++i) {
        GpuMat gm(137, 240*32, DataType<float>::type);
        DevMem2D_<float> dm = gm;
        features.push_back(dm);
        features_gm.push_back(gm);
    }

    for (int j = 0; j < nfilters; ++j) {
        GpuMat gm(4, 4*32, DataType<float>::type);
        DevMem2D_<float> dm = gm;
        filters.push_back(dm);
        filters_gm.push_back(gm);
    }

    for (int i = 0; i < nfeatures; ++i) {
        vector<DevMem2D_<float> > response_level;
        for (int j = 0; j < nfilters; ++j) {
            //GpuMat gm(137-5+1, (240-5+1), DataType<float>::type);
            GpuMat gm(137-4+1, 240-4+1, DataType<float>::type);
            DevMem2D_<float> dm = gm;
            response_level.push_back(dm);
        }
        responses.push_back(response_level);
    }

    // convolve the filters with the features
    double t = (double)cv::getTickCount();
    fconv(features, filters, responses);
    std::cout << "My convolution time: " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << std::endl;

    // convolve using standard OpenCV methods
    ConvolveBuf buf;
    Stream stream;
    t = (double)cv::getTickCount();
    for (int i = 0; i < nfeatures; ++i) {
        for (int j = 0; j < nfilters; ++j) {
            GpuMat response;
            cv::gpu::convolve(features_gm[i], filters_gm[j], response, false, buf, stream);
        }
    }
    stream.waitForCompletion();
    std::cout << "OpenCV convolution time: " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << std::endl;

    return 0;
}
