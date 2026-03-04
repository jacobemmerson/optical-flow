cmake_minimum_required(VERSION 3.22)
project(DenseFlowTest LANGUAGES CXX CUDA)

# Use C++17 (avoids GCC 13 C++20 issues)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Path to your LibTorch
set(CMAKE_PREFIX_PATH /home/tae/libs/libtorch) # adjust if different

find_package(Torch REQUIRED)

# Enable CUDA if you have GPU support
find_package(CUDA)

# Your executable
add_executable(denseflow dense_flow.cpp)

# Link LibTorch
target_link_libraries(denseflow "${TORCH_LIBRARIES}")

# Include directories
target_include_directories(denseflow PUBLIC
    ${TORCH_INCLUDE_DIRS}
)

# Avoid RPATH issues on Linux
set_target_properties(denseflow PROPERTIES
    BUILD_RPATH "${TORCH_LIB_DIR}"
)