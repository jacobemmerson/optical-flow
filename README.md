# Lucas-Kanade Sparse and Dense Optical Flow with Fused Kernels
Tae Emmerson | Jan. 24, 2026

This project implements the classical Lucas-Kanade optical flow algorithm as a single fused CUDA kernel and benchmarks it against OpenCV's GPU sparse implementation. Both run single-level with a 7x7 window and no pyramids.

I started from the common assumption that Lucas-Kanade is memory-bound, but Nsight Compute told a different story. On this workload DRAM sits near 6 percent of peak and both kernels are cache-resident, so bytes moved is not the bottleneck. The custom kernel is faster because it does less work: it maps one warp (32 threads) to each tracked point and reduces the structure tensor with warp shuffles, while OpenCV assigns a full 256-thread block per point. OpenCV executes roughly 3.7x more instructions to compute the same flow, and the custom kernel matches its accuracy to within 0.2 px.

https://github.com/user-attachments/assets/a146e0d6-aaae-44ab-bbaf-8f32d3134031

## Results

*Measured on a GTX 1650 Ti and Intel i7 over 200 KITTI Optical Flow pairs, 7x7 window, single-level (no pyramids). GPU times are warm and device-timed with CUDA events.*

### Throughput

| Implementation | Avg time per frame (ms) | Hardware |
| --- | --- | --- |
| Python (OpenCV) | 0.739740 | CPU |
| C++ (OpenCV) | 0.837934 | CPU |
| C++ (OpenCV) | 0.304002 | GPU |
| CUDA (fused kernel) | 0.179676 | GPU |

The fused kernel is **1.69x faster** than OpenCV's GPU implementation.

### Accuracy vs KITTI ground truth

Each arm is scored only on the points it keeps (min-eigenvalue rejection, matching OpenCV's `minEigThreshold`) that also have valid ground truth, so the comparison is like-for-like. Density is the share of valid-ground-truth points each arm kept.

| Implementation | Mean EPE (px) | Median EPE (px) | Fl-outliers (%) | Density (%) |
| --- | --- | --- | --- | --- |
| CUDA (fused kernel) | 35.31 | 21.09 | 79.2 | 99.8 |
| C++ (OpenCV, GPU) | 35.13 | 20.89 | 73.6 | 99.4 |

The custom kernel matches OpenCV to within 0.2 px on both mean and median EPE, which confirms it is a faithful reimplementation rather than a fast approximation. The absolute error is high because single-level Lucas-Kanade cannot capture KITTI's large inter-frame motions; both implementations hit the same algorithmic ceiling by design (`maxLevel = 0`). This benchmark measures kernel efficiency and correctness parity, not state-of-the-art accuracy, which would require a pyramidal implementation.

### Why it is faster

Nsight Compute counters (clocks locked for comparability):

| Metric | CUDA (fused) | OpenCV (GPU) |
| --- | --- | --- |
| DRAM throughput (% of peak) | 6.1 | 1.4 |
| L1 hit rate (%) | 96 | 94 |
| Instructions executed | 3.42M | 12.65M |

The workload is not memory-bound: DRAM stays far below peak and both kernels are cache-resident. The speedup comes from thread-to-work mapping. The fused kernel assigns one warp per point; OpenCV assigns a 256-thread block per point plus a separate helper kernel, so it executes about 3.7x more instructions for the same result. Same bytes, same accuracy, far less work.

## Build

Building requires CUDA 13 and an OpenCV built with CUDA support, plus a small
patch to a CUDA system header on recent glibc. See [SETUP.md](SETUP.md) for the
full reproducible steps.

# Optical Flow

*Note: The work below was transcribed by ChatGPT into code blocks since GitHub doesn't render multi-line math equations. My original work can be found in `math.md`*

Optical flow algorithms require two **assumptions**:

### 1. Brightness Constancy

For a point moving through an image sequence, the brightness of the point will remain the same.

```
I(x(t), y(t), t) = C
```

### 2. Small Motion

For a very small space-time step, the brightness between two consecutive image frames is the same.

```
I(x + uΔt, y + vΔt, t + Δt) = I(x, y, t)
```

Combined, these assumptions yield the **brightness constancy equation**:

```
dI/dt = ∂I/∂x * dx/dt + ∂I/∂y * dy/dt + ∂I/∂t = 0
```

In layman's terms, the change in brightness `∂I/∂x` and `∂I/∂y`, caused by motion `dx/dt` and `dy/dt`, must cancel the observed brightness change over time `∂I/∂t`.

We can derive the equation from the assumptions.

```
I(x + uΔt, y + vΔt, t + Δt) = I(x, y, t)
```

Using a first-order Taylor expansion:

```
I(x,y,t) + ∂I/∂x Δx + ∂I/∂y Δy + ∂I/∂t Δt = I(x,y,t)
```

Cancel terms:

```
∂I/∂x Δx + ∂I/∂y Δy + ∂I/∂t Δt = 0
```

Divide by `Δt` and take the limit `Δt → 0`:

```
∂I/∂x dx/dt + ∂I/∂y dy/dt + ∂I/∂t = 0
```

Shorthand notation:

```
Ix u + Iy v + It = 0
```

Vector form:

```
∇Iᵀ v + It = 0
```

**Goal:** In optical flow, we want to solve for `u` and `v`.
But we have **two unknowns and one equation**.

## Lucas–Kanade (Sparse)

### Assumptions

1. Flow is locally smooth
2. Neighboring pixels have the same displacement

Consider a `5 × 5` image patch with constant flow. This patch gives us **25 equations**:

```
Ix(p_i)u + Iy(p_i)v = -It(p_i)    for i = 1 … 25
```

Matrix form:

```
[ Ix(p1)  Iy(p1) ]
[ Ix(p2)  Iy(p2) ]
[   ...     ...  ]
[ Ix(p25) Iy(p25) ]  [u]  =  -[ It(p1) ]
                      [v]     [ It(p2) ]
                              [  ...   ]
                              [ It(p25)]
```

We solve this using **least squares**:

```
x = (AᵀA)⁻¹ Aᵀ b
```

Which yields:

```
[ Σ IxIx   Σ IxIy ] [u] = -[ Σ IxIt ]
[ Σ IyIx   Σ IyIy ] [v]   [ Σ IyIt ]
```

## Citations

- https://www.cs.cmu.edu/~16385/s15/lectures/Lecture21.pdf
