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

private:
    std::vector<cv::Mat> img1s, img2s, gray1s, gray2s;
};