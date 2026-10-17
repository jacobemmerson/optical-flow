# Lucas-Kanade Sparse Optical Flow in CUDA

A single fused CUDA kernel for Lucas-Kanade sparse optical flow, benchmarked against OpenCV's GPU
implementation (`cv::cuda::SparsePyrLKOpticalFlow`). Both run single-level (no pyramid) with a 7x7
window and 10 iterations.

The difference from OpenCV is thread-to-work mapping. The kernel assigns one warp (32 threads) to
each tracked point and reduces the structure tensor with warp shuffles. OpenCV assigns a 256-thread
block per point and launches a separate helper kernel.

https://github.com/user-attachments/assets/a146e0d6-aaae-44ab-bbaf-8f32d3134031

## Results

200 KITTI 2015 training pairs, up to 2,000 `goodFeaturesToTrack` points per frame, GTX 1650 Ti and
Intel i7.

### Speed

| implementation | ms per frame | hardware |
| --- | --- | --- |
| Python (OpenCV) | 0.740 | CPU |
| C++ (OpenCV) | 0.838 | CPU |
| C++ (OpenCV) | 0.304 | GPU |
| CUDA (this kernel) | 0.180 | GPU |

The kernel is 1.69x faster than OpenCV on the GPU. GPU times are CUDA-event timings around the
flow computation only (no host transfers or feature detection), after one warm-up run per frame
(`cuda/main.cpp`). CPU times use `std::chrono` (`cuda/main_cpu.cpp`) and Python's timer
(`python/main.py`).

### Accuracy against KITTI ground truth

Each arm is scored on the points it keeps (min-eigenvalue rejection, matching OpenCV's
`minEigThreshold`) that also have valid ground truth. Density is the share of valid-ground-truth
points each arm kept.

| implementation | mean EPE (px) | median EPE (px) | Fl-outliers (%) | density (%) |
| --- | --- | --- | --- | --- |
| CUDA (this kernel) | 35.31 | 21.09 | 79.2 | 99.8 |
| C++ (OpenCV, GPU) | 35.13 | 20.89 | 73.6 | 99.4 |

Mean and median EPE are within 0.2 px of OpenCV. The Fl-outlier rate (EPE > 3 px and > 5% of the
true flow) is 5.6 points higher; the cause has not been investigated. Absolute error is high for
both because single-level Lucas-Kanade cannot track KITTI's large motions; a pyramid would be
needed for competitive accuracy. This benchmark measures speed and parity with OpenCV.

### Why it is faster

Nsight Compute, one steady-state launch per arm, clocks locked (`scripts/profile_ncu.sh`, default 1200 MHz):

| metric | CUDA (this kernel) | OpenCV (GPU) |
| --- | --- | --- |
| DRAM throughput (% of peak) | 6.1 | 1.4 |
| L1 hit rate (%) | 96 | 94 |
| instructions executed | 3.42M | 12.65M |

The workload is not memory-bound: DRAM use is far below peak and both kernels hit L1 over 90% of
the time. OpenCV executes 3.7x more instructions for the same flow, which accounts for the speedup.

`bench/results/lk_counters.{csv,md}` are from an earlier capture that did not record instructions
executed; rerun `scripts/profile_ncu.sh` to regenerate them with that counter.

## Build and run

Requires CUDA 13 and OpenCV 4.14 built from source with CUDA and the contrib `cudaoptflow` module.
On glibc 2.43 the CUDA math header also needs a one-line patch. [SETUP.md](SETUP.md) has the full
steps.

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
cmake -S cuda -B cuda/build      # change CUDA_ARCHITECTURES "75" in cuda/CMakeLists.txt for non-Turing GPUs
cmake --build cuda/build
```

Download KITTI 2015 scene flow (`data/README.md`) so that `data/training/image_2/` and
`data/training/flow_occ/` exist. The binaries use a relative data path, so run them from
`cuda/build`:

```bash
cd cuda/build
./lk_gpu      # kernel vs OpenCV GPU: timing, speedup, KITTI accuracy
./lk_cpu      # kernel vs OpenCV CPU
./bird        # video demo
cd ../.. && scripts/profile_ncu.sh   # Nsight Compute counters (locks clocks with sudo)
```

Python baselines (sparse Lucas-Kanade and Farneback dense flow through OpenCV):

```bash
cd python && python main.py --iterations 10 --measure 2
```

## Background

Lucas-Kanade assumes brightness constancy and small motion, which give one equation per pixel,
`Ix u + Iy v + It = 0`, with two unknowns. Assuming constant flow over a window gives an
overdetermined system solved by least squares:

```
[ Σ IxIx   Σ IxIy ] [u]     [ Σ IxIt ]
[ Σ IxIy   Σ IyIy ] [v] = - [ Σ IyIt ]
```

The kernel iterates this solve 10 times per point and rejects points whose structure tensor's
minimum eigenvalue is below the threshold. The full derivation is in [math.md](math.md).

Reference: https://www.cs.cmu.edu/~16385/s15/lectures/Lecture21.pdf
