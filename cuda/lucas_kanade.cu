#include <cuda_runtime.h>

#define PATCH_RADIUS 3
#define PATCH_SIZE (2*PATCH_RADIUS+1)
#define PATCH_PIXELS (PATCH_SIZE*PATCH_SIZE)
#define MAX_ITERS 10

__device__ inline float bilinear(
    const float* img,
    int w,
    int h,
    float x,
    float y)
{
    int x0 = floorf(x);
    int y0 = floorf(y);

    float dx = x - x0;
    float dy = y - y0;

    x0 = max(0,min(w-2,x0));
    y0 = max(0,min(h-2,y0));

    float I00 = img[y0*w+x0];
    float I01 = img[y0*w+x0+1];
    float I10 = img[(y0+1)*w+x0];
    float I11 = img[(y0+1)*w+x0+1];

    return (1-dx)*(1-dy)*I00 +
           dx*(1-dy)*I01 +
           (1-dx)*dy*I10 +
           dx*dy*I11;
}

__device__ inline float warpReduce(float val)
{
    for(int offset=16; offset>0; offset/=2)
        val += __shfl_down_sync(0xffffffff,val,offset);

    return val;
}

__global__ void lk_warp_kernel(
    const float* prev,
    const float* curr,
    const float2* pts,
    float2* flow,
    int n,
    int w,
    int h)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= n) return;

    float2 p = pts[warp_id];

    float u = 0.f;
    float v = 0.f;

    for(int iter=0; iter<MAX_ITERS; iter++)
    {
        float Gxx=0,Gyy=0,Gxy=0;
        float bx=0,by=0;

        if (lane < PATCH_PIXELS)
        {
            int dx = lane % PATCH_SIZE - PATCH_RADIUS;
            int dy = lane / PATCH_SIZE - PATCH_RADIUS;

            float x = p.x + dx;
            float y = p.y + dy;

            float I1 = bilinear(prev,w,h,x,y);
            float I2 = bilinear(curr,w,h,x+u,y+v);

            float Ix =
                bilinear(prev,w,h,x+1,y) -
                bilinear(prev,w,h,x-1,y);

            float Iy =
                bilinear(prev,w,h,x,y+1) -
                bilinear(prev,w,h,x,y-1);

            float It = I2 - I1;

            Gxx = Ix*Ix;
            Gyy = Iy*Iy;
            Gxy = Ix*Iy;

            bx = Ix*It;
            by = Iy*It;
        }

        Gxx = warpReduce(Gxx);
        Gyy = warpReduce(Gyy);
        Gxy = warpReduce(Gxy);

        bx = warpReduce(bx);
        by = warpReduce(by);

        if (lane == 0)
        {
            float det = Gxx*Gyy - Gxy*Gxy;

            if (fabs(det) > 1e-6f)
            {
                float inv = 1.f/det;

                float du = (-Gyy*bx + Gxy*by)*inv;
                float dv = ( Gxy*bx - Gxx*by)*inv;

                u += du;
                v += dv;

                if (du*du + dv*dv < 1e-4)
                    iter = MAX_ITERS;
            }
        }

        u = __shfl_sync(0xffffffff,u,0);
        v = __shfl_sync(0xffffffff,v,0);
    }

    if (lane == 0)
        flow[warp_id] = make_float2(u,v);
}

void run_lucas_kanade_cuda(
    const float* d_prev,
    const float* d_curr,
    const float2* d_points,
    float2* d_flow,
    int n_points,
    int width,
    int height)
{
    int warps_per_block = 8;
    int threads = warps_per_block * 32;

    int blocks = (n_points + warps_per_block - 1) / warps_per_block;

    lk_warp_kernel<<<blocks,threads>>>(
        d_prev,
        d_curr,
        d_points,
        d_flow,
        n_points,
        width,
        height);
}