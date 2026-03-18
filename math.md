# Lucas-Kanade Sparse and Dense Optical Flow with Fused Kernels

Tae Emmerson | Jan. 24, 2026

## Introduction

This project explores the throughput of the classical Lucas-Kanade optical flow algorithm based on its OpenCV (C++/Python) and CUDA implementation from scratch. 

### Optical Flow:

Optical flow algorithms require two **assumptions**:
1. **Brightness Constancy**: For a point moving through an image sequence, the brightness of the point will remain the same.
$$
I(x(t), y(t), t) = C
$$
2. **Small Motion**: For a really (really) small space-time step, the brightness between two consecutive image frames is the same.
$$
I(x + u \delta t, y + v \delta t, t + \delta t) = I(x, y, t)
$$

Combined, theese assumptions yield the **brightness constancy equation**:
$$
\frac{dI}{dt} = \frac{\partial I}{\partial x} \frac{dx}{dt} + \frac{\partial I}{\partial y} \frac{dy}{dt} + \frac{\partial I}{\partial t} = 0
$$

In laymen's terms, the change in brightness $\frac{\partial I}{\partial x}$ and $\frac{\partial I}{\partial y}$, caused by motion, $\frac{dx}{dt}$ and $\frac{dy}{dt}$, must cancel the observed brightness change over time $\frac{\partial I}{\partial t}$.

We can derive the equation from the assumptions.
$$
\begin{align*}
I(x + u \delta t, y + v \delta t, t + \delta t) &= I(x, y, t) \:\: \text{assuming small motion, expand taylor series}\\
I(x,y,t) + \frac{\partial I}{\partial x} \delta x + \frac{\partial I}{\partial y} \delta y + \frac{\partial I}{\partial t} \delta t &= I(x,y,t) \:\: \text{cancel terms}  \\
\frac{\partial I}{\partial x} \delta x + \frac{\partial I}{\partial y} \delta y + \frac{\partial I}{\partial t} \delta t &= 0 \:\: \text{divide by } \delta t \text{ and take limit } \delta t \rightarrow 0 \\
\frac{\partial I}{\partial x} \frac{dx}{dt} + \frac{\partial I}{\partial y} \frac{dy}{dt} + \frac{\partial I}{\partial t} &= 0 \:\: \text{\textbf{Brightness Constancy Equation}} \\
I_x u + I_y v + I_t &= 0 \:\: \text{shorthand notation} \\
\nabla I^Tv + I_t &= 0 \:\: \text{vector form}
\end{align*}
$$

**Goal**: in optical flow, we want to solve for $u$ and $v$! But we have two unknowns and one equation... what can we do?

### Lucas-Kanade (Sparse)

Assumptions:
1. Flow is locally smooth
2. Neighboring pixels have the same displacement

Let's consider a $5 \times 5$ image patch with constant flow. This patch gives us $25$ equations:
$$
I_x(p_i)u + I_y(p_i)v = -I_t(p_i) \text{ for } i = 1\dots25
$$
or
$$
\begin{bmatrix}
    I_x(p_1) & I_y(p_1) \\
    I_x(p_2) & I_y(p_2) \\
    \vdots & \vdots \\
    I_x(p_{25}) & I_y(p_{25}) \\
\end{bmatrix}

\begin{bmatrix}
    u \\
    v
\end{bmatrix}
=
-
\begin{bmatrix}
I_t(p_1) \\
I_t(p_2) \\
\vdots \\
I_t(p_{25})
\end{bmatrix}
$$

We can solve this using least squares approximation $x = (A^TA)^{-1}A^Tb$:

$$
\begin{bmatrix}
\sum_{p \in P} I_x I_x & \sum_{p \in P}I_x I_y \\
\sum_{p \in P} I_y I_x & \sum_{p \in P}I_y I_y \\
\end{bmatrix}

\begin{bmatrix}
    u \\
    v
\end{bmatrix}

=
-
\begin{bmatrix}
\sum_{p \in P} I_x I_t \\
\sum_{p \in P} I_y I_t
\end{bmatrix}
$$

## Citations

- https://www.cs.cmu.edu/~16385/s15/lectures/Lecture21.pdf
