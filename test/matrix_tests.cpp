#include "matrix.hpp" // your header file with the implementations
#include <gtest/gtest.h>
#include <random>
#include <vector>

class MatrixMultiplicationTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // Fill matrices with random values
    for (auto &val : left)
      val = dis(gen);
    for (auto &val : right)
      val = dis(gen);
    // Initialize result matrices with zeros
    result_naive.assign(Rows * Columns, 0.0f);
    result_loop_order.assign(Rows * Columns, 0.0f);
    result_accelerate.assign(Rows * Columns, 0.0f);
  }

  static constexpr int Rows = 4;
  static constexpr int Columns = 3;
  static constexpr int Inners = 2;

  std::vector<float> left{std::vector<float>(Rows * Inners)};
  std::vector<float> right{std::vector<float>(Inners * Columns)};
  std::vector<float> result_naive{std::vector<float>(Rows * Columns)};
  std::vector<float> result_loop_order{std::vector<float>(Rows * Columns)};
  std::vector<float> result_accelerate{std::vector<float>(Rows * Columns)};

  // Helper function to compare two float vectors with tolerance
  bool compareResults(const std::vector<float> &a, const std::vector<float> &b,
                      float tolerance = 1e-5f) {
    if (a.size() != b.size())
      return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (std::abs(a[i] - b[i]) > tolerance)
        return false;
    }
    return true;
  }

  // Helper to print matrix for debugging
  void printMatrix(const std::vector<float> &matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        std::cout << matrix[i * cols + j] << " ";
      }
      std::cout << std::endl;
    }
  }
};

TEST_F(MatrixMultiplicationTest, CompareDifferentImplementations) {
  // Run all implementations
  matmulImplNaive<Rows, Columns, Inners>(left.data(), right.data(),
                                         result_naive.data());

  matmulImplLoopOrder<Rows, Columns, Inners>(left.data(), right.data(),
                                             result_loop_order.data());

  matmulImplAccelerate<Rows, Columns, Inners>(left.data(), right.data(),
                                              result_accelerate.data());

  // Compare results
  ASSERT_TRUE(compareResults(result_naive, result_loop_order))
      << "Naive and Loop Order implementations produce different results";

  ASSERT_TRUE(compareResults(result_naive, result_accelerate))
      << "Naive and Accelerate implementations produce different results";

  // If test fails, print matrices for debugging
  if (::testing::Test::HasFailure()) {
    std::cout << "\nLeft Matrix:" << std::endl;
    printMatrix(left, Rows, Inners);

    std::cout << "\nRight Matrix:" << std::endl;
    printMatrix(right, Inners, Columns);

    std::cout << "\nNaive Result:" << std::endl;
    printMatrix(result_naive, Rows, Columns);

    std::cout << "\nLoop Order Result:" << std::endl;
    printMatrix(result_loop_order, Rows, Columns);

    std::cout << "\nAccelerate Result:" << std::endl;
    printMatrix(result_accelerate, Rows, Columns);
  }
}

// Test with known values
TEST_F(MatrixMultiplicationTest, KnownValues) {
  // Override random values with known values
  std::vector<float> known_left = {1, 2, // 2x2 matrix
                                   3, 4};
  std::vector<float> known_right = {5, 6, // 2x2 matrix
                                    7, 8};
  std::vector<float> expected = {19, 22, // Expected result
                                 43, 50};

  std::vector<float> result(4, 0.0f);

  // Test all implementations with known values
  matmulImplAccelerate<2, 2, 2>(known_left.data(), known_right.data(),
                                result.data());

  ASSERT_TRUE(compareResults(result, expected))
      << "Accelerate implementation failed for known values";
}

// Test edge cases
TEST_F(MatrixMultiplicationTest, EdgeCases) {
  // Test with 1x1 matrices
  std::vector<float> left_1x1 = {2.0f};
  std::vector<float> right_1x1 = {3.0f};
  std::vector<float> result_1x1(1, 0.0f);
  std::vector<float> expected_1x1 = {6.0f};

  matmulImplAccelerate<1, 1, 1>(left_1x1.data(), right_1x1.data(),
                                result_1x1.data());

  ASSERT_TRUE(compareResults(result_1x1, expected_1x1))
      << "Failed 1x1 matrix multiplication";
}
