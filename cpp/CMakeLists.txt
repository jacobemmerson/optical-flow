cmake_minimum_required(VERSION 3.22)
project(SparseFlowTest LANGUAGES CXX CUDA)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)


# Your executable
add_executable(sparseflow sparse_flow.cpp ../utils/utils.cpp)
add_executable(denseflow dense_flow.cpp ../utils/utils.cpp)

# Link LibTorch
target_link_libraries(sparseflow ${TORCH_LIBRARIES})
target_link_libraries(sparseflow ${OpenCV_LIBS})
target_link_libraries(denseflow ${TORCH_LIBRARIES})
target_link_libraries(denseflow ${OpenCV_LIBS})

# Include directories
target_include_directories(sparseflow PUBLIC
    ${TORCH_INCLUDE_DIRS}
)
target_include_directories(denseflow PUBLIC
    ${TORCH_INCLUDE_DIRS}
)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Avoid RPATH issues on Linux
set_target_properties(sparseflow PROPERTIES
    BUILD_RPATH "${TORCH_LIB_DIR}"
)
set_target_properties(denseflow PROPERTIES
    BUILD_RPATH "${TORCH_LIB_DIR}"
)