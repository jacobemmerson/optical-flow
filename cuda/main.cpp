#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include "utils/utils.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define MAX_FEATURES 2000

// CUDA kernel launcher
void run_lucas_kanade_cuda(
    const float* d_prev,
    const float* d_curr,
    const float2* d_points,
    float2* d_flow,
    unsigned char* d_status,
    int n_points,
    int width,
    int height);

// KITTI-style sparse accuracy for one arm. Each arm is scored only where it
// keeps a point (status == 1) and ground truth is valid, so mean EPE compares
// like flow against like; density is the share of valid-GT points it kept.
struct AccuracyStats {
    std::vector<float> epes; // per-point EPE for scored points, for the median
    double epe_sum = 0;
    long outliers = 0;  // KITTI Fl: EPE > 3px and EPE > 5% of |GT|
    long gt_valid = 0;  // GT valid, kept or not -> density denominator
};

void scorePoint(
    AccuracyStats& acc,
    const cv::Mat& groundTruth,
    const cv::Point2f& point,
    float du,
    float dv,
    bool kept)
{
    int x = std::lround(point.x);
    int y = std::lround(point.y);
    if (x < 0 || y < 0 || x >= groundTruth.cols || y >= groundTruth.rows) return;

    cv::Vec3f gt = groundTruth.at<cv::Vec3f>(y, x);
    if (gt[2] == 0.f) return; // no ground-truth flow at this pixel

    acc.gt_valid++;
    if (!kept) return;

    float ex = du - gt[0];
    float ey = dv - gt[1];
    float epe = std::sqrt(ex*ex + ey*ey);

    acc.epe_sum += epe;
    acc.epes.push_back(epe);

    float magnitude = std::sqrt(gt[0]*gt[0] + gt[1]*gt[1]);
    if (epe > 3.f && epe > 0.05f * magnitude) acc.outliers++;
}

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
        "{ plot | | visualize flow }"
        "{ profile | | capture one warm frame between cudaProfiler"
        " start/stop for Nsight Compute, then exit }");

    bool plot = parser.has("plot");
    bool profile = parser.has("profile");

    FlowSet dataset("../../data", true);

    int WIDTH = 0;
    int HEIGHT = 0;

    float *d_prev = nullptr;
    float *d_curr = nullptr;

    float2 *d_points = nullptr;
    float2 *d_flow = nullptr;

    float2 h_points[MAX_FEATURES];
    float2 h_flow[MAX_FEATURES];
    unsigned char h_status[MAX_FEATURES];

    unsigned char *d_status = nullptr;

    bool first_frame = true;

    double cuda_total = 0;
    double opencv_total = 0;

    // KITTI accuracy accumulators. Each arm is scored only on points it keeps
    // (status == 1) that also have valid ground truth, so mean EPE is
    // like-for-like; density reveals how aggressively each arm rejects.
    AccuracyStats cuda_acc, opencv_acc;

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
            cudaMalloc(&d_status, MAX_FEATURES * sizeof(unsigned char));

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

        // ===== Prepare both implementations =====

        // OpenCV GPU inputs
        cv::cuda::GpuMat d_img1(img1_gray);
        cv::cuda::GpuMat d_img2(img2_gray);
        cv::cuda::GpuMat d_pts1(pts1);
        cv::cuda::GpuMat d_pts2;
        cv::cuda::GpuMat d_cv_status;
        cv::cuda::GpuMat d_err;

        cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_lk = cv::cuda::SparsePyrLKOpticalFlow::create(
            cv::Size(7,7), // winsize
            0, // max level = 0, single level no pyramid
            10 // max iters to match my kernel
        );

        // Both arms are warmed before timing. In profile mode extra warmups
        // ensure the captured launch is steady-state (caches/autotune settled).
        int warmup_iters = profile ? 10 : 1;
        for (int w = 0; w < warmup_iters; ++w)
        {
            run_lucas_kanade_cuda(d_prev,d_curr,d_points,d_flow,d_status,n_points,WIDTH,HEIGHT);
            d_lk->calc(d_img1,d_img2,d_pts1,d_pts2,d_cv_status,d_err);
        }
        cudaDeviceSynchronize();

        // ===== Timed region =====
        // Both arms are device-timed with cudaEvents so the comparison is
        // like-for-like. Under Nsight Compute (`--profile`) these event times
        // are meaningless -- ncu replays kernels -- so this stdout timing is
        // only valid on a normal run; the ncu run is for counters alone.

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        if (profile) cudaProfilerStart();

        // Custom fused kernel
        cudaEventRecord(start);
        run_lucas_kanade_cuda(d_prev,d_curr,d_points,d_flow,d_status,n_points,WIDTH,HEIGHT);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float cuda_ms;
        cudaEventElapsedTime(&cuda_ms, start, stop);
        cuda_total += cuda_ms;

        cudaMemcpy(h_flow, d_flow,
            n_points * sizeof(float2),
            cudaMemcpyDeviceToHost);

        cudaMemcpy(h_status, d_status,
            n_points * sizeof(unsigned char),
            cudaMemcpyDeviceToHost);

        // OpenCV GPU LK
        cudaEventRecord(start);
        d_lk->calc(d_img1,d_img2,d_pts1,d_pts2,d_cv_status,d_err);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float opencv_ms;
        cudaEventElapsedTime(&opencv_ms, start, stop);
        opencv_total += opencv_ms;

        if (profile) cudaProfilerStop();

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // ===== Accuracy vs KITTI ground truth =====
        if (!profile)
        {
            cv::Mat groundTruth = dataset.groundTruthFlow(i);

            cv::Mat pts2(d_pts2);
            cv::Mat cv_status(d_cv_status);

            for (int k = 0; k < n_points; ++k)
            {
                scorePoint(cuda_acc, groundTruth, pts1[k],
                           h_flow[k].x, h_flow[k].y, h_status[k] != 0);

                cv::Point2f moved = pts2.at<cv::Point2f>(k);
                scorePoint(opencv_acc, groundTruth, pts1[k],
                           moved.x - pts1[k].x, moved.y - pts1[k].y,
                           cv_status.at<uchar>(k) != 0);
            }
        }

        // ===== VISUALIZATION ===== (debugging)

        if (plot)
        {
            cv::Mat display = sample.img2.clone();
            cv::Mat pts2(d_pts2);
            cv::Mat status(d_cv_status);
            drawSparseFlow(pts1, pts2, status, display);

            cv::imshow("OpenCV Sparse LK", display);

            if (cv::waitKey(1) == 27)
                break;
        }

        if (profile) break;
    }

    size_t frame_count = profile ? 1 : dataset.size();
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

    if (!profile)
    {
        auto report = [](const char* name, AccuracyStats& acc) {
            long scored = static_cast<long>(acc.epes.size());
            double mean = scored ? acc.epe_sum / scored : 0.0;
            double median = 0.0;
            if (scored) {
                auto mid = acc.epes.begin() + scored / 2;
                std::nth_element(acc.epes.begin(), mid, acc.epes.end());
                median = *mid;
            }
            double outlier_pct = scored ? 100.0 * acc.outliers / scored : 0.0;
            double density = acc.gt_valid ? 100.0 * scored / acc.gt_valid : 0.0;
            std::cout << name
                      << "  mean EPE: " << mean << " px"
                      << " | median EPE: " << median << " px"
                      << " | Fl-outliers: " << outlier_pct << "%"
                      << " | density: " << density << "%"
                      << " (" << scored << "/" << acc.gt_valid << ")\n";
        };

        std::cout << "\nAccuracy vs KITTI ground truth:\n";
        report("CUDA  ", cuda_acc);
        report("OpenCV", opencv_acc);
    }

    cudaFree(d_prev);
    cudaFree(d_curr);
    cudaFree(d_points);
    cudaFree(d_flow);
    cudaFree(d_status);

    return 0;
}