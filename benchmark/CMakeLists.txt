add_executable(matrix_benchmark
    matrix_benchmark.cpp
)

target_link_libraries(matrix_benchmark
    PRIVATE
    matrix_lib
    benchmark::benchmark
    benchmark::benchmark_main
)

# Set compiler flags for benchmarking
target_compile_options(matrix_benchmark PRIVATE
    -O3
    -march=native
    -DNDEBUG
)
