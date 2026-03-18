# Lucas-Kanade Sparse and Dense Optical Flow with Fused Kernels
Tae Emmerson | Jan. 24, 2026

This project explores the throughput of the classical Lucas-Kanade optical flow algorithm based on its OpenCV (C++/Python) and CUDA implementation from scratch. As Lucas-Kanade is a memory-bound algorithm, particularly on modest GPUs, I explore kernel fusion to reduce the overhead of memory operations on a basic implementation of the algorithm that meets the standards of OpenCV's with no Pyramids.

<div align="center">
<video src="data/birds.mp4" width="700" loop autoplay muted playsinline>
  Your browser does not support the video tag.
</video>
</div>

https://github.com/user-attachments/assets/a146e0d6-aaae-44ab-bbaf-8f32d3134031



## Results

*These results were produced using a GTX 1650ti and Intel i7 on the KITTI Optical Flow dataset with a 7x7 window, single-level (no pyramids).*

| Language | Average Time per Frame (ms) | Hardware|
| --- | --- | --- |
| Python | 0.739740 | CPU |
| C++ | 0.837934 | CPU |
| C++ | 0.558732 | GPU |
| CUDA | 0.180362 | GPU |

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
