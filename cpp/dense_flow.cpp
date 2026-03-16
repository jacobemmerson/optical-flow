#include <iostream>
#include "../utils/utils.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <chrono>
#include <vector>

// Add this helper function somewhere in main.cpp (e.g. before main(), or inside main before the loop)
cv::Mat visualizeDenseFlow(const cv::Mat& flow, double mag_scale = 10.0) {
    // flow is CV_32FC2
    std::vector<cv::Mat> flow_split(2);
    cv::split(flow, flow_split);

    cv::Mat mag, ang;
    cv::cartToPolar(flow_split[0], flow_split[1], mag, ang, true);  // true = degrees (0..360)

    // Convert to HSV for nice color coding
    ang.convertTo(ang, CV_8U, 0.5);                    // 0..180 (hue range)
    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX, CV_8U);  // scale magnitude nicely

    std::vector<cv::Mat> hsv_channels = {
        ang,
        cv::Mat::ones(ang.size(), CV_8U) * 255,   // full saturation
        mag
    };

    cv::Mat hsv, bgr;
    cv::merge(hsv_channels, hsv);
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr;
}

int main(int argc, char** argv) {
    // === Command line (kept only the plot flag) ===
    cv::CommandLineParser parser(argc, argv,
        "{ h help |      | print this help message }"
        "{ @plot  |      | show optical flow tracks }");
    parser.about("Fast Lucas-Kanade on KITTI 2012/2015");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    bool plot = parser.has("@plot");

    // Random colors for visualization
    std::vector<cv::Scalar> colors;
    colors.reserve(100);
    cv::RNG rng;
    for (int i = 0; i < 100; ++i) {
        colors.emplace_back(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    }

    FlowSet dataset("../../data", true);
    cv::Mat img1_gray, img2_gray;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::Mat flow;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < dataset.size(); ++i) {
        auto sample = dataset.get(i);

        cv::cvtColor(sample.img1, img1_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(sample.img2, img2_gray, cv::COLOR_BGR2GRAY);

        cv::calcOpticalFlowFarneback(
            img1_gray,
            img2_gray,
            flow,
            0.5, // pyr_scale
            2, // levels
            15, // winsize
            3, // iterations
            5, // poly_n
            1.5, // poly_sigma
            0 // flags
        );

        if (plot) {
                    cv::Mat flow_vis = visualizeDenseFlow(flow);

                    // Blend with original image so you can still see the scene
                    cv::Mat display;
                    cv::addWeighted(sample.img2, 0.7, flow_vis, 0.3, 0.0, display);

                    cv::imshow("Dense Farneback Optical Flow", display);
                    int k = cv::waitKey(1);
                    if (k == 'q' || k == 27) break;
                }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time running: " << duration.count() / 1e6 << " seconds\n";

    return 0;
}