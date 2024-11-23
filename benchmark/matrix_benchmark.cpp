#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#include "matrix.hpp"

std::vector<float> initialize_random_vector(size_t size) {
  static thread_local std::random_device rd;
  static thread_local std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  std::vector<float> result(size);
  std::generate(result.begin(), result.end(), [&]() { return dis(gen); });
  return result;
}

static void BM_MatmulNaive(benchmark::State &state) {
  std::vector<float> left = initialize_random_vector(rows * inners);
  std::vector<float> right = initialize_random_vector(inners * columns);
  std::vector<float> result(rows * columns);

  for (auto _ : state) {
    std::fill(result.begin(), result.end(), 0.0f);
    matmulImplNaive<rows, columns, inners>(left.data(), right.data(),
                                           result.data());
    benchmark::DoNotOptimize(result.data());
  }

  state.SetItemsProcessed(state.iterations() * rows * columns * inners);
  state.SetLabel(OPTIMIZATION_LEVEL);
}

static void BM_MatmulLoopOrder(benchmark::State &state) {
  std::vector<float> left = initialize_random_vector(rows * inners);
  std::vector<float> right = initialize_random_vector(inners * columns);
  std::vector<float> result(rows * columns);

  for (auto _ : state) {
    std::fill(result.begin(), result.end(), 0.0f);
    matmulImplLoopOrder<rows, columns, inners>(left.data(), right.data(),
                                               result.data());
    benchmark::DoNotOptimize(result.data());
  }

  state.SetItemsProcessed(state.iterations() * rows * columns * inners);
  state.SetLabel(OPTIMIZATION_LEVEL);
}

static void BM_MatmulAccelerate(benchmark::State &state) {
  std::vector<float> left = initialize_random_vector(rows * inners);
  std::vector<float> right = initialize_random_vector(inners * columns);
  std::vector<float> result(rows * columns);

  for (auto _ : state) {
    std::fill(result.begin(), result.end(), 0.0f);
    matmulImplAccelerate<rows, columns, inners>(left.data(), right.data(),
                                                result.data());
    benchmark::DoNotOptimize(result.data());
  }

  state.SetItemsProcessed(state.iterations() * rows * columns * inners);
  state.SetLabel(OPTIMIZATION_LEVEL);
}

BENCHMARK(BM_MatmulNaive)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MatmulLoopOrder)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MatmulAccelerate)->Unit(benchmark::kMillisecond);
