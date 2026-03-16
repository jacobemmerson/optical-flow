#include <iostream>
#include "../utils/utils.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <chrono>
#include <vector>

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

    cv::Mat mask;                     // reused if plotting

    cv::Mat img1_gray, img2_gray;
    std::vector<cv::Point2f> prevPts, nextPts;
    std::vector<uchar> status;
    std::vector<float> err;

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < dataset.size(); ++i) {
        auto sample = dataset.get(i);

        cv::cvtColor(sample.img1, img1_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(sample.img2, img2_gray, cv::COLOR_BGR2GRAY);

        // Reset mask every frame (pairs are independent!)
        if (plot) {
            mask = cv::Mat::zeros(sample.img1.size(), sample.img1.type());
        }

        // Detect features only on first image
        prevPts.clear();
        cv::goodFeaturesToTrack(img1_gray, prevPts, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);

        if (prevPts.empty()) continue;

        // Prepare nextPts (let LK start from prevPts positions)
        nextPts = prevPts;

        cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.03);
        cv::calcOpticalFlowPyrLK(img1_gray, img2_gray,
                                 prevPts, nextPts, status, err,
                                 cv::Size(15, 15), 2, criteria);

        // Build good_new + draw (only if plotting)
        if (plot) {
            for (size_t j = 0; j < prevPts.size(); ++j) {
                if (status[j] == 1) {
                    cv::line(mask, nextPts[j], prevPts[j], colors[j], 2);
                    cv::circle(sample.img2, nextPts[j], 5, colors[j], -1);  // note: sample.img2 is const ref but we don't modify the stored one
                }
            }
            cv::Mat display;
            cv::add(sample.img2, mask, display);
            cv::imshow("Optical Flow", display);
            int k = cv::waitKey(1);
            if (k == 'q' || k == 27) break;
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time running: " << duration.count() / 1e6 << " seconds\n";

    return 0;
}