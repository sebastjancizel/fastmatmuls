#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#include "matrix.hpp"

constexpr int rows = 1024;
constexpr int columns = 1024;
constexpr int inners = 1024;

std::vector<float> initialize_random_vector(size_t size) {
  static thread_local std::random_device rd;
  static thread_local std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  std::vector<float> result(size);
  std::generate(result.begin(), result.end(), [&]() { return dis(gen); });
  return result;
}

template <MatmulImplementation auto func>
static void BM_Matmul(benchmark::State &state) {
  std::vector<float> left = initialize_random_vector(rows * inners);
  std::vector<float> right = initialize_random_vector(inners * columns);
  std::vector<float> result(rows * columns);

  for (auto _ : state) {
    std::fill(result.begin(), result.end(), 0.0f);
    func(rows, columns, inners, left.data(), right.data(), result.data());
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * rows * columns * inners);
}

BENCHMARK(BM_Matmul<matmulImplAccelerate>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Matmul<matmulImplNaive>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Matmul<matmulImplLoopOrder>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Matmul<matmulImplTiling<16>>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Matmul<matmulImplTiling<18>>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Matmul<matmulImplTiling<32>>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Matmul<matmulImplTiling<128>>)->Unit(benchmark::kMillisecond);
