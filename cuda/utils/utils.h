#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <tuple>

namespace fs = std::filesystem;

struct FlowSample {
    const cv::Mat& img1;      // BGR
    const cv::Mat& img2;      // BGR
    const cv::Mat& img1_gray; // CV_8UC1
    const cv::Mat& img2_gray; // CV_8UC1
};

class FlowSet {
public:
    FlowSet(const std::string& root, bool /*occ*/ = true); // occ parameter kept for API compatibility

    size_t size() const { return img1s.size(); }
    FlowSample get(size_t index) const {
        return {img1s[index], img2s[index], gray1s[index], gray2s[index]};
    }

    // Decodes the KITTI flow_occ map for a sample into a CV_32FC3 image whose
    // channels are (u, v, valid). Loaded on demand -- the encoded 16-bit maps
    // are ~2.8 MB each, so preloading all 200 would waste hundreds of MB.
    cv::Mat groundTruthFlow(size_t index) const;

private:
    std::vector<cv::Mat> img1s, img2s, gray1s, gray2s;
    std::vector<std::string> flow_paths;
};