cmake_minimum_required(VERSION 3.30)
project(matmul VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define optimization configurations as separate lists
set(OPT_NAMES    "O0"    "O1"   "O3"    "Ofast")
set(OPT_FLAGS    "-O0"   "-O1"   "-O3"   "-Ofast")
set(OPT_DESC     "No optimizations" "Basic optimizations" "Aggressive optimizations" "Fast math optimizations")

# Find required packages
find_package(GTest REQUIRED)
find_package(Benchmark REQUIRED)

# Base library
add_library(matrix_lib INTERFACE)
add_compile_definitions(ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
target_include_directories(matrix_lib INTERFACE include)
target_link_libraries(matrix_lib INTERFACE "-framework Accelerate")

# Create benchmarks for each optimization level
list(LENGTH OPT_NAMES NUM_CONFIGS)
math(EXPR NUM_CONFIGS "${NUM_CONFIGS} - 1")

foreach(IDX RANGE ${NUM_CONFIGS})
    list(GET OPT_NAMES ${IDX} CONFIG_NAME)
    list(GET OPT_FLAGS ${IDX} COMPILER_FLAG)
    
    add_executable(matrix_benchmark_${CONFIG_NAME} benchmark/matrix_benchmark.cpp)
    target_link_libraries(matrix_benchmark_${CONFIG_NAME} PRIVATE
        matrix_lib
        benchmark::benchmark
        benchmark::benchmark_main
    )
    target_compile_options(matrix_benchmark_${CONFIG_NAME} PRIVATE ${COMPILER_FLAG})
    
    # Add description as a compile definition
    target_compile_definitions(matrix_benchmark_${CONFIG_NAME} PRIVATE 
      OPTIMIZATION_LEVEL="${CONFIG_NAME}"
    )
endforeach()

enable_testing()

add_executable(
  matrix_test
  ./test/matrix_tests.cpp
)
target_link_libraries(
  matrix_test
  matrix_lib
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(matrix_test)
