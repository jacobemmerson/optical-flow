#include <cuda_runtime.h>
#include <stdio.h>

// Device helper: solve 2x2 system with singularity + eigenvalue checks
__device__ inline float2 solve_lk_system(float sxx, float sxy, float syy, float sxt, float syt)
{
    const float DET_THRESHOLD   = 1.5e-5f;
    const float EIGEN_THRESHOLD = 1e-3f;

    float det = sxx * syy - sxy * sxy;
    if (fabsf(det) < DET_THRESHOLD) return make_float2(0.0f, 0.0f);

    float trace = sxx + syy;
    float twice_delta = sqrtf(trace * trace - 4.0f * det);
    if (isnan(twice_delta) || (trace - twice_delta) <= EIGEN_THRESHOLD) {
        return make_float2(0.0f, 0.0f);
    }

    float flow_x = (syy * (-sxt) + sxy * syt) / det;
    float flow_y = (sxy * sxt - sxx * syt) / det;

    return make_float2(flow_x, flow_y);
}

// Kernel 1: Gradients
__global__ void compute_gradients_kernel(
    const float* __restrict__ prev,
    const float* __restrict__ curr,
    float* __restrict__ Ix,
    float* __restrict__ Iy,
    float* __restrict__ It,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    It[idx] = curr[idx] - prev[idx];

    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        Ix[idx] = (prev[y*width + x+1] - prev[y*width + x-1]) * 0.5f;
        Iy[idx] = (prev[(y+1)*width + x] - prev[(y-1)*width + x]) * 0.5f;
    } else {
        Ix[idx] = Iy[idx] = 0.0f;
    }
}

// Kernel 2: Fused Lucas-Kanade 
__global__ void lucas_kanade_fused_kernel(
    const float* __restrict__ Ix,
    const float* __restrict__ Iy,
    const float* __restrict__ It,
    float* __restrict__ flow_u,
    float* __restrict__ flow_v,
    int width, int height, int win_half)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sxx = 0.0f, sxy = 0.0f, syy = 0.0f, sxt = 0.0f, syt = 0.0f;

    for (int dy = -win_half; dy <= win_half; ++dy) {
        for (int dx = -win_half; dx <= win_half; ++dx) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = ny * width + nx;
                float fx = Ix[nidx], fy = Iy[nidx], ft = It[nidx];
                sxx += fx * fx;
                syy += fy * fy;
                sxy += fx * fy;
                sxt += fx * ft;
                syt += fy * ft;
            }
        }
    }

    float2 flow = solve_lk_system(sxx, sxy, syy, sxt, syt);
    int idx = y * width + x;
    flow_u[idx] = flow.x;
    flow_v[idx] = flow.y;
}

// Launcher (to be called each frame)
void run_lucas_kanade_cuda(
    const float* d_prev,
    const float* d_curr,
    float* d_flow_u,
    float* d_flow_v,
    int width,
    int height,
    int window_size)
{
    int win_half = window_size / 2;

    float *d_Ix, *d_Iy, *d_It;
    cudaMalloc(&d_Ix, width * height * sizeof(float));
    cudaMalloc(&d_Iy, width * height * sizeof(float));
    cudaMalloc(&d_It, width * height * sizeof(float));

    dim3 block(16, 16);
    dim3 grid((width + 15)/16, (height + 15)/16);

    compute_gradients_kernel<<<grid, block>>>(d_prev, d_curr, d_Ix, d_Iy, d_It, width, height);
    lucas_kanade_fused_kernel<<<grid, block>>>(d_Ix, d_Iy, d_It, d_flow_u, d_flow_v, width, height, win_half);

    cudaFree(d_Ix); cudaFree(d_Iy); cudaFree(d_It);
    cudaDeviceSynchronize();  // safe during development
}