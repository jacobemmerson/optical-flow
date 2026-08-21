# Build and Setup

This project builds a custom CUDA Lucas-Kanade kernel and benchmarks it against
OpenCV's GPU implementation, so it needs a CUDA toolkit and an OpenCV that was
built with CUDA support. Neither comes from a single package manager step. This
document reproduces the environment from scratch.

## Reference environment

The numbers in the README were produced on this configuration:

| Component | Version |
| --- | --- |
| OS | Ubuntu 26.04 (glibc 2.43) |
| GPU | GTX 1650 Ti (Turing, `sm_75`), driver 580 |
| CUDA | 13.0.88 |
| OpenCV | 4.14.0 + contrib, built with CUDA |
| Nsight Compute | 2024.1.1 (system `ncu`) |

If your GPU is not Turing, change `CUDA_ARCHITECTURES "75"` in
`cuda/CMakeLists.txt` to your compute capability (for example `86` for Ampere).

## 1. Install CUDA 13

Install the individual CUDA components, not the `cuda-toolkit-13-0`
metapackage. On Ubuntu 26.04 the metapackage pulls in
`cuda-nsight-systems-13-0`, which cannot satisfy its `nsight-systems`
dependency and leaves apt in a broken state.

```bash
sudo apt-get install cuda-nvcc-13-0 cuda-cudart-dev-13-0 \
                     cuda-libraries-dev-13-0 cuda-nvrtc-dev-13-0
```

Add the toolkit to your PATH (needed for every build and profile command):

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
```

Profiling uses the system `ncu` (`/usr/bin/ncu`, 2024.1). It is older than
CUDA 13 but reads Turing counters correctly. Do not install the CUDA 13 Nsight
packages; that reintroduces the apt conflict above.

## 2. Patch the CUDA math header (required on glibc 2.43)

With CUDA 13.0.88 on glibc 2.43, no `.cu` file compiles out of the box. nvcc
fails with:

```
exception specification is incompatible with that of previous function "rsqrt"
```

glibc 2.43 declares `rsqrt`/`rsqrtf` as `noexcept` (under `_GNU_SOURCE`, which
both nvcc and OpenCV set), while CUDA 13.0.88 declares them without it. Add
`noexcept` to the two declarations in the CUDA header:

File: `/usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h`

```
double rsqrt(double x) noexcept;
float  rsqrtf(float x) noexcept;
```

Back up the header first so the change is reversible:

```bash
sudo cp /usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h{,.bak}
```

Caveat: a CUDA package update overwrites this header and silently reintroduces
the break. Every build (this project and OpenCV) fails again until you re-apply
the patch. No newer nvcc package fixes it; 13.0.88 is the current candidate.

## 3. Build OpenCV with CUDA

No apt package ships `cv::cuda::SparsePyrLKOpticalFlow`; it lives in the contrib
`cudaoptflow` module and must be built from source with CUDA.

```bash
git clone --depth 1 --branch 4.14.0 https://github.com/opencv/opencv.git
git clone --depth 1 --branch 4.14.0 https://github.com/opencv/opencv_contrib.git

cmake -S opencv -B opencv/build \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D OPENCV_EXTRA_MODULES_PATH=$(pwd)/opencv_contrib/modules \
  -D WITH_CUDA=ON \
  -D CUDA_ARCH_BIN=7.5

cmake --build opencv/build -j6
sudo cmake --build opencv/build --target install
```

Notes from this build:
- If you restrict modules with `BUILD_LIST`, you must include `cudev`;
  configuration fails without it because the CUDA modules depend on it.
- Use `-j6` on a 16 GB machine. The CUDA translation units are memory-hungry
  and `-j12` risks the OOM killer.
- highgui was built without GTK, so `imshow` does not display. The `--plot`
  flag on `lk_gpu` and the `bird` video demo will not show a window; the
  benchmark and accuracy paths do not need it.

## 4. Build this project

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
cmake -S cuda -B cuda/build
cmake --build cuda/build
```

This produces three binaries in `cuda/build`:
- `lk_gpu`: the benchmark (custom kernel vs OpenCV GPU) and accuracy evaluation
- `lk_cpu`: the custom kernel vs OpenCV's CPU implementation
- `bird`: a video demo of the custom kernel

## 5. Get the dataset

The KITTI 2015 scene-flow data is not in the repo. Download it (see
`data/README.md`) and extract so the layout is:

```
data/training/image_2/*_10.png   # first frame
data/training/image_2/*_11.png   # second frame
data/training/flow_occ/*_10.png  # ground-truth flow
```

## 6. Run

The binaries resolve the dataset with a relative path, so run them from their
own directory:

```bash
cd cuda/build
./lk_gpu            # prints per-frame timing, speedup, and KITTI accuracy
```

To capture Nsight Compute counters (locks GPU clocks with sudo, resets on exit):

```bash
scripts/profile_ncu.sh    # writes bench/results/lk_profile.ncu-rep and a table
```

See the script header for options (`CLOCK`, `LOCK_CLOCKS`).
