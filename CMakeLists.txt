cmake_minimum_required(VERSION 3.30)
# Use LLVM's Clang for OpenMP support
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(matmul VERSION 0.1.0 LANGUAGES CXX)

# Add optimization flags
add_compile_options(-march=native -ffast-math)

enable_testing()

# Find required packages
find_package(OpenMP REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)

# Base library
add_library(matrix_lib INTERFACE)
add_compile_definitions(ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
target_include_directories(matrix_lib INTERFACE include)
target_link_libraries(matrix_lib INTERFACE
    "-framework Accelerate"
    OpenMP::OpenMP_CXX
    GTest::gtest
    benchmark::benchmark
)

add_subdirectory(benchmark)

option(testing_enabled "Enable testing" ON)
if(testing_enabled)
  add_subdirectory(test)
endif()
