#include <iostream>
#include "../utils/utils.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <chrono>
#include <cuda_runtime.h>

#define MAX_FEATURES 2000

// CUDA kernel launcher
void run_lucas_kanade_cuda(
    const float* d_prev,
    const float* d_curr,
    const float2* d_points,
    float2* d_flow,
    int n_points,
    int width,
    int height);

// Draw sparse flow
void drawSparseFlow(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const std::vector<uchar>& status,
    cv::Mat& img)
{
    for (size_t i = 0; i < pts1.size(); i++)
    {
        if (!status[i]) continue;

        cv::arrowedLine(img,
            pts1[i],
            pts2[i],
            cv::Scalar(0,255,0),
            1,8,0,0.3);
    }
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv,
        "{ h help | | help }"
        "{ plot | | visualize flow }");

    bool plot = parser.has("plot");

    FlowSet dataset("../../data", true);

    int WIDTH = 0;
    int HEIGHT = 0;

    float *d_prev = nullptr;
    float *d_curr = nullptr;

    float2 *d_points = nullptr;
    float2 *d_flow = nullptr;

    float2 h_points[MAX_FEATURES];
    float2 h_flow[MAX_FEATURES];

    bool first_frame = true;

    double cuda_total = 0;
    double opencv_total = 0;

    for (size_t i = 0; i < dataset.size(); ++i)
    {
        auto sample = dataset.get(i);

        if (first_frame)
        {
            WIDTH = sample.img1.cols;
            HEIGHT = sample.img1.rows;

            cudaMalloc(&d_prev, WIDTH * HEIGHT * sizeof(float));
            cudaMalloc(&d_curr, WIDTH * HEIGHT * sizeof(float));

            cudaMalloc(&d_points, MAX_FEATURES * sizeof(float2));
            cudaMalloc(&d_flow,   MAX_FEATURES * sizeof(float2));

            first_frame = false;
        }

        // grayscale conversion
        cv::Mat img1_gray, img2_gray;
        cv::cvtColor(sample.img1, img1_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(sample.img2, img2_gray, cv::COLOR_BGR2GRAY);

        cv::Mat prev_float, curr_float;
        img1_gray.convertTo(prev_float, CV_32F);
        img2_gray.convertTo(curr_float, CV_32F);

        // upload images
        cudaMemcpy(d_prev, prev_float.ptr<float>(),
            WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(d_curr, curr_float.ptr<float>(),
            WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

        // ===== Feature Detection =====

        std::vector<cv::Point2f> pts1;

        cv::goodFeaturesToTrack(
            img1_gray,
            pts1,
            MAX_FEATURES,
            0.01,
            8);

        int n_points = pts1.size();

        for (int k = 0; k < n_points; k++)
        {
            h_points[k].x = pts1[k].x;
            h_points[k].y = pts1[k].y;
        }

        cudaMemcpy(d_points, h_points,
            n_points * sizeof(float2),
            cudaMemcpyHostToDevice);

        // ===== CUDA Fused Kernel Implementation =====

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        run_lucas_kanade_cuda(
            d_prev,
            d_curr,
            d_points,
            d_flow,
            n_points,
            WIDTH,
            HEIGHT);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float cuda_ms;
        cudaEventElapsedTime(&cuda_ms, start, stop);

        cuda_total += cuda_ms;

        cudaMemcpy(h_flow, d_flow,
            n_points * sizeof(float2),
            cudaMemcpyDeviceToHost);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // ===== OpenCV LK Implementation =====

        std::vector<cv::Point2f> pts2;
        std::vector<uchar> status;
        std::vector<float> err;

        auto t1 = std::chrono::high_resolution_clock::now();

        cv::calcOpticalFlowPyrLK(
            img1_gray,
            img2_gray,
            pts1,
            pts2,
            status,
            err,
            cv::Size(7,7),
            1);

        auto t2 = std::chrono::high_resolution_clock::now();

        double opencv_ms =
            std::chrono::duration<double,std::milli>(t2 - t1).count();

        opencv_total += opencv_ms;

        // ===== VISUALIZATION ===== (debugging)

        if (plot)
        {
            cv::Mat display = sample.img2.clone();

            drawSparseFlow(pts1, pts2, status, display);

            cv::imshow("OpenCV Sparse LK", display);

            if (cv::waitKey(1) == 27)
                break;
        }
    }

    auto frame_count = dataset.size();
    std::cout << "\nFrames processed: " << frame_count << "\n";

    std::cout << "\nCUDA LK avg time: "
              << cuda_total / frame_count
              << " ms\n";

    std::cout << "OpenCV LK avg time: "
              << opencv_total / frame_count
              << " ms\n";

    std::cout << "\nSpeedup: "
              << (opencv_total / cuda_total)
              << "x\n";

    cudaFree(d_prev);
    cudaFree(d_curr);
    cudaFree(d_points);
    cudaFree(d_flow);

    return 0;
}