#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include "utils/utils.h"

#define MAX_FEATURES 500
#define decay 0.85f

// CUDA kernel launcher
void run_lucas_kanade_cuda(
    const float* d_prev,
    const float* d_curr,
    const float2* d_points,
    float2* d_flow,
    int n_points,
    int width,
    int height);

int main()
{
    cv::VideoCapture capture("../../data/bird.mp4");
    if (!capture.isOpened()) {
        std::cerr << "Unable to open video." << std::endl;
        return -1;
    }

    int WIDTH = 0, HEIGHT = 0;
    float *d_prev = nullptr, *d_curr = nullptr;
    float2 *d_points = nullptr, *d_flow = nullptr;

    float2 h_points[MAX_FEATURES];
    float2 h_flow[MAX_FEATURES];

    cv::Mat old_frame;
    capture >> old_frame;
    if (old_frame.empty()) return -1;

    // Create mask outside the loop (same as before)
    cv::Mat mask = cv::Mat::zeros(old_frame.size(), old_frame.type());

    WIDTH = old_frame.cols;
    HEIGHT = old_frame.rows;

    // Allocate GPU memory
    cudaMalloc(&d_prev, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc(&d_curr, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc(&d_points, MAX_FEATURES * sizeof(float2));
    cudaMalloc(&d_flow, MAX_FEATURES * sizeof(float2));

    cv::VideoWriter video("../../birds.avi",cv::VideoWriter::fourcc('M','J','P','G'), 30, cv::Size(WIDTH, HEIGHT),true);

    while (true)
    {
        cv::Mat frame;
        capture >> frame;
        if (frame.empty()) break;

        // Convert to grayscale and float
        cv::Mat old_gray, frame_gray;
        cv::cvtColor(old_frame, old_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

        cv::Mat prev_float, curr_float;
        old_gray.convertTo(prev_float, CV_32F);
        frame_gray.convertTo(curr_float, CV_32F);

        // Upload to GPU
        cudaMemcpy(d_prev, prev_float.ptr<float>(), WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_curr, curr_float.ptr<float>(), WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

        // Feature detection
        std::vector<cv::Point2f> pts1;
        cv::goodFeaturesToTrack(
            old_gray, // input array
            pts1,  // output array
            MAX_FEATURES, 
            0.3, // quality level
            0 // min distance
        );

        int n_points = pts1.size();
        if (n_points == 0) {
            old_frame = frame.clone();
            continue;
        }

        // Copy points to GPU
        for (int k = 0; k < n_points; k++) {
            h_points[k].x = pts1[k].x;
            h_points[k].y = pts1[k].y;
        }
        cudaMemcpy(d_points, h_points, n_points * sizeof(float2), cudaMemcpyHostToDevice);

        // Warmup kernel
        run_lucas_kanade_cuda(d_prev, d_curr, d_points, d_flow, 1, WIDTH, HEIGHT);
        cudaDeviceSynchronize();

        // Run fused kernel
        run_lucas_kanade_cuda(d_prev, d_curr, d_points, d_flow, n_points, WIDTH, HEIGHT);
        cudaDeviceSynchronize();

        // Download flow
        cudaMemcpy(h_flow, d_flow, n_points * sizeof(float2), cudaMemcpyDeviceToHost);

        // Build pts2 and status
        std::vector<cv::Point2f> pts2(n_points);
        std::vector<uchar> status(n_points, 1); // assume all points valid
        for (int k = 0; k < n_points; k++)
            pts2[k] = cv::Point2f(h_points[k].x + h_flow[k].x,
                                   h_points[k].y + h_flow[k].y);


        // Fade the existing mask
        mask *= decay;  // element-wise multiply, old trails fade
        // Draw new tracks on top of the decayed mask
        for (int k = 0; k < n_points; k++) {
            //cv::line(mask, pts2[k], pts1[k], cv::Scalar(255, 0, 255), 1, 1);
            cv::circle(mask, pts2[k], 2, cv::Scalar(255, 0, 255), 4);
        }

        cv::Mat display;
        cv::addWeighted(frame, 1, mask, 0.3, 0, display);
        video.write(display);
        cv::imshow("CUDA Sparse LK Video", display);
        int key = cv::waitKey(1);
        if (key == 27) break;

        old_frame = frame.clone(); // advance frame
    }

    cudaFree(d_prev);
    cudaFree(d_curr);
    cudaFree(d_points);
    cudaFree(d_flow);

    return 0;
}