# Matrix Multiplication Project

Implementation of the [Fast Multidimensional Matrix Multiplication on CPU from Scratch](https://simonboehm.de/blog/2024/01/25/fast-multidimensional-matrix-multiplication-on-cpu-from-scratch/) blog post by Simon Boehm. After reading the blog post, I was curious to see things look on my Macbook Pro.

## Prerequisites

- CMake (>= 3.30)
- C++20 compatible compiler
- Conan package manager
- macOS with Accelerate framework

## Installing CMake and Conan

I recommend using [uv](https://docs.astral.sh/uv/) to use CMake and Conan. You can install as a tool using `uv tool install` or even just run it with `uvx`.

## Building the Project

1. Install dependencies using Conan:
```bash
conan install . --output-folder=build --build=missing
```

2. Create and navigate to the build directory:
```bash
cd build
```

3. Configure the project with CMake:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```
By default, compile commands are exported to `compile_commands.json` in the build directory. They are useful for tools like [clangd](https://clangd.llvm.org/) and [VSCode](https://code.visualstudio.com/).

4. Build all targets:
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

## VSCode/Neovim Configuration

I recommend using clangd for a the language server. The Microsoft C/C++ extension has issues with the Accelerate framework.

## Project Structure

- `include/` - Header files
- `src/` - Source files
- `benchmark/` - Benchmark implementation
- `test/` - Test files