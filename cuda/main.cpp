#include <iostream>
#include "../utils/utils.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>

// === CUDA includes and function declaration ===
#include <cuda_runtime.h>

// Declare the fused kernel launcher (defined in lucas_kanade.cu)
void run_lucas_kanade_cuda(
    const float* d_prev,
    const float* d_curr,
    float* d_flow_u,
    float* d_flow_v,
    int width,
    int height,
    int window_size = 7);

// Dense flow visualization helper (green arrows, every 16 pixels)
void drawDenseFlow(const cv::Mat& flow_u, const cv::Mat& flow_v, cv::Mat& img, int step = 16)
{
    for (int y = 0; y < img.rows; y += step) {
        for (int x = 0; x < img.cols; x += step) {
            float ux = flow_u.at<float>(y, x);
            float vy = flow_v.at<float>(y, x);
            float mag = std::sqrt(ux * ux + vy * vy);
            if (mag < 1.0f) continue;               // skip tiny motion

            cv::Point p1(x, y);
            cv::Point p2(cvRound(x + ux), cvRound(y + vy));
            cv::arrowedLine(img, p1, p2, cv::Scalar(0, 255, 0), 1, 8, 0, 0.3f);
        }
    }
}

int main(int argc, char** argv) {
    // === Command line (kept only the plot flag) ===
    cv::CommandLineParser parser(argc, argv,
        "{ h help |     | print this help message }"
        "{ @plot  |     | show optical flow (dense arrows) }");
    parser.about("Fast Fused CUDA Lucas-Kanade on KITTI 2012/2015");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    bool plot = parser.has("@plot");

    // === Dataset ===
    FlowSet dataset("../../data", true);

    // === Persistent device buffers (allocated once) ===
    int WIDTH = 0, HEIGHT = 0;
    float *d_prev = nullptr, *d_curr = nullptr;
    float *d_u = nullptr, *d_v = nullptr;
    cv::Mat flow_u_host, flow_v_host;
    bool first_frame = true;

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < dataset.size(); ++i) {
        auto sample = dataset.get(i);

        // First frame → allocate everything
        if (first_frame) {
            WIDTH  = sample.img1.cols;
            HEIGHT = sample.img1.rows;

            flow_u_host.create(HEIGHT, WIDTH, CV_32F);
            flow_v_host.create(HEIGHT, WIDTH, CV_32F);

            cudaMalloc(&d_prev, WIDTH * HEIGHT * sizeof(float));
            cudaMalloc(&d_curr, WIDTH * HEIGHT * sizeof(float));
            cudaMalloc(&d_u,    WIDTH * HEIGHT * sizeof(float));
            cudaMalloc(&d_v,    WIDTH * HEIGHT * sizeof(float));

            first_frame = false;
        }

        // Convert to grayscale + float (0-255 range — kernel handles it)
        cv::Mat img1_gray, img2_gray, prev_float, curr_float;
        cv::cvtColor(sample.img1, img1_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(sample.img2, img2_gray, cv::COLOR_BGR2GRAY);
        img1_gray.convertTo(prev_float, CV_32F);
        img2_gray.convertTo(curr_float, CV_32F);

        // Upload to GPU
        cudaMemcpy(d_prev, prev_float.ptr<float>(), WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_curr, curr_float.ptr<float>(), WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

        // === THIS IS THE FUSED CUDA KERNEL CALL (replaces calcOpticalFlowPyrLK) ===
        run_lucas_kanade_cuda(d_prev, d_curr, d_u, d_v, WIDTH, HEIGHT, 7);

        // Download dense flow
        cudaMemcpy(flow_u_host.ptr<float>(), d_u, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(flow_v_host.ptr<float>(), d_v, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

        // === Visualization (dense arrows instead of sparse tracks) ===
        if (plot) {
            cv::Mat display = sample.img2.clone();
            drawDenseFlow(flow_u_host, flow_v_host, display);

            cv::imshow("Dense CUDA Optical Flow", display);
            int k = cv::waitKey(1);
            if (k == 'q' || k == 27) break;
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time running: " << duration.count() / 1e6 << " seconds\n";

    // Cleanup
    if (d_prev) cudaFree(d_prev);
    if (d_curr) cudaFree(d_curr);
    if (d_u)    cudaFree(d_u);
    if (d_v)    cudaFree(d_v);

    return 0;
}