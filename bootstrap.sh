#!/bin/bash
set -e  # Exit on error

# Check if brew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew is required but not installed. Please install it first."
    exit 1
fi

# Install LLVM if not present
if ! brew list llvm &> /dev/null; then
    echo "Installing LLVM..."
    brew install llvm
fi

echo "Setting up Conan profile..."
conan profile detect --force

echo "Installing dependencies with Conan..."
conan install . --output-folder=build --build=missing

echo "Configuring with CMake..."
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building..."
cmake --build .

echo "Build complete! You can now run the benchmarks with:"
echo "./benchmark/matrix_benchmark"