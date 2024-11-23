#include <benchmark/benchmark.h>
#include <vector>

#include "matrix.hpp"

constexpr int rows = 1024;
constexpr int columns = 1024;
constexpr int inners = 1024;

static constexpr std::vector<float> initialize_random_vector(int n) {
  std::vector<float> result(rows * inners);
  for (auto &val : result)
    val = static_cast<float>(rand()) / RAND_MAX;
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
