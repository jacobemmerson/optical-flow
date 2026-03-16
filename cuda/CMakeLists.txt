cmake_minimum_required(VERSION 3.22)
project(fast_lk LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)           # modern way (works on CUDA 10+)
enable_language(CUDA)

add_executable(
    fast_lk
    main.cpp
    ../utils/utils.cpp
    lucas_kanade.cu
)

target_include_directories(fast_lk PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(fast_lk PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
)

target_compile_options(fast_lk PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

set_target_properties(fast_lk PROPERTIES
    CUDA_ARCHITECTURES "75"          
    CUDA_SEPARABLE_COMPILATION OFF
)

# --- Nice build output ---
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")  

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA version:   ${CUDAToolkit_VERSION}")