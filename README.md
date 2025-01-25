# Matrix Multiplication Project

Implementation of the amazing [Fast Multidimensional Matrix Multiplication on CPU from Scratch](https://simonboehm.de/blog/2024/01/25/fast-multidimensional-matrix-multiplication-on-cpu-from-scratch/) blog post by Simon Boehm. After reading the blog post, I was curious to see things look on my Mac and how close the techniques used in the blog post are to the ones used in the [Accelerate framework](https://developer.apple.com/documentation/accelerate). I also wanted to create a self-contained playground, complete with testing and benchmarking to quickly iterate on different implementations.

## Prerequisites

- CMake (>= 3.30)
- C++20 compatible compiler
- Conan package manager
- macOS with Accelerate framework

## Installing CMake and Conan

I recommend using [uv](https://docs.astral.sh/uv/) to use CMake and Conan. You can install as a tool using `uv tool install` or even just run it with `uvx`.

## Building the Project

You can either use the bootstrap script:
```bash
./bootstrap.sh
```

Or follow these steps manually:

1. Setup Conan profile (first time only):
```bash
conan profile detect --force
```

2. Install dependencies using Conan:
```bash
conan install . --output-folder=build --build=missing
```
This will install the dependencies and generate the `conanbuildinfo.txt` file.

3. Create and navigate to the build directory:
```bash
cd build
```

4. Configure the project with CMake:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```
By default, compile commands are exported to `compile_commands.json` in the build directory. They are useful for tools like [clangd](https://clangd.llvm.org/) and [VSCode](https://code.visualstudio.com/).

5. Build all targets:
```bash
cmake --build .
```

### Building Specific Targets

- To build just the benchmarks:
```bash
cmake --build . --target matrix_benchmark
```

- To build and run tests:
```bash
cmake --build . --target test
```

## Running Benchmarks

After building, you can run the benchmarks from the build directory:
```bash
./benchmark/matrix_benchmark
```

### Results
On my Macbook Pro M2 with 16GB RAM, the results are as follows:
```
--------------------------------------------------------------------------------------------------------
Benchmark                                              Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------------------------
BM_Matmul<matmulImplAccelerate>                    0.866 ms        0.829 ms          830 items_per_second=1.29449T/s
BM_Matmul<matmulImplNaive>                           915 ms          913 ms            1 items_per_second=1.17642G/s
BM_Matmul<matmulImplLoopOrder>                      77.9 ms         77.7 ms            9 items_per_second=13.8165G/s
BM_Matmul<matmulImplTiling<8>>                      70.5 ms         70.3 ms           10 items_per_second=15.2707G/s
BM_Matmul<matmulImplTiling<16>>                     68.4 ms         68.2 ms           10 items_per_second=15.7465G/s
BM_Matmul<matmulImplTiling<18>>                     66.9 ms         66.7 ms           10 items_per_second=16.0875G/s
BM_Matmul<matmulImplTilingRowCol<256,256,16>>       14.2 ms         10.3 ms           70 items_per_second=104.343G/s
```

## VSCode/Neovim Configuration

I recommend using clangd for a the language server. The Microsoft C/C++ extension has issues with the Accelerate framework.

## Project Structure

- `include/` - Header files
- `benchmark/` - Benchmark implementation
- `test/` - Test files