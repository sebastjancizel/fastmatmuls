#pragma once

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <omp.h>
#include <arm_neon.h>  // Required for ARM NEON intrinsics

/*
 * The MatmulImplementation concept defines the requirements for any function that implements
 * matrix multiplication. It ensures that any function satisfying this concept:
 *
 * 1. Takes exactly 6 parameters in the following order:
 *    - rows: number of rows in the result matrix (and left matrix)
 *    - cols: number of columns in the result matrix (and right matrix)
 *    - inner: number of columns in left matrix (also rows in right matrix)
 *    - left: pointer to the left matrix data (rows × inner)
 *    - right: pointer to the right matrix data (inner × cols)
 *    - result: pointer to store the resulting matrix (rows × cols)
 *
 * 2. Returns void (specified by std::same_as<void>)
 *
 * This concept is used to ensure type safety and consistent interfaces across different
 * matrix multiplication implementations, allowing them to be used interchangeably
 * while guaranteeing they follow the same parameter convention.
 *
 * In benchmarking, this concept enables generic benchmark templates that can test any
 * compliant implementation. By using 'template <MatmulImplementation auto func>', we can
 * create a single benchmark function that works with any matrix multiplication implementation
 * that satisfies the concept.
 */
template <typename F>
concept MatmulImplementation =
    requires(F f, int rows, int cols, int inner, const float *left,
             const float *right, float *result) {
      { f(rows, cols, inner, left, right, result) } -> std::same_as<void>;
    };

inline void matmulImplNaive(int rows, int columns, int inners,
                            const float *left, const float *right,
                            float *result) {
  for (int row = 0; row < rows; row++) {
    for (int column = 0; column < columns; column++) {
      float sum = 0.0f;
      for (int inner = 0; inner < inners; inner++) {
        sum += left[row * inners + inner] * right[inner * columns + column];
      }
      result[row * columns + column] = sum;
    }
  }
}

inline void matmulImplLoopOrder(int rows, int columns, int inners,
                                const float *left, const float *right,
                                float *result) {
  std::fill_n(result, rows * columns, 0.0f);
  for (int row = 0; row < rows; row++) {
    for (int inner = 0; inner < inners; inner++) {
      for (int column = 0; column < columns; column++) {
        result[row * columns + column] +=
            left[row * inners + inner] * right[inner * columns + column];
      }
    }
  }
}

inline void matmulImplAccelerate(int rows, int columns, int inners,
                                 const float *left, const float *right,
                                 float *result) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, columns, inners,
              1.0f, left, inners, right, columns, 0.0f, result, columns);
}

template <int TileSize>
inline void matmulImplTiling(int rows, int columns, int inners,
                             const float *left, const float *right,
                             float *result) {
  std::fill_n(result, rows * columns, 0.0f);
  for (int innerTile = 0; innerTile < inners; innerTile += TileSize) {
    for (int row = 0; row < rows; row++) {
      int innerTileEnd = std::min(inners, innerTile + TileSize);
      for (int inner = innerTile; inner < innerTileEnd; inner++) {
        for (int column = 0; column < columns; column++) {
          result[row * columns + column] +=
              left[row * inners + inner] * right[inner * columns + column];
        }
      }
    }
  }
}

template <int RowTileSize, int ColumnTileSize, int InnerTileSize>
/*
 * 3D tiling for matrix multiplication with OpenMP parallelization:
 * - Tiles the output matrix into RowTileSize × ColumnTileSize blocks
 * - For each output tile, processes the inner dimension in InnerTileSize chunks
 *
 * Parallelization strategy:
 * 1. Parallel region spans rowTile and columnTile loops (collapse(2))
 *    - Increases parallel granularity
 *    - Better load balancing across threads
 *    - Each thread processes multiple complete tiles
 *
 * 2. Data sharing:
 *    - result/left/right arrays are shared between threads
 *    - Loop indices and bounds are private by default
 *    - default(none) ensures explicit declaration of all variables
 *
 * Cache optimization:
 * - Row tiling ensures each thread works on a cache-friendly slice
 * - Column tiling improves spatial locality in the output matrix
 * - Inner dimension tiling reduces cache misses in matrix traversal
 */
inline void matmulImplTilingRowCol(int rows, int columns, int inners,
                                  const float *left, const float *right,
                                  float *result) {
  std::fill_n(result, rows * columns, 0.0f);

  #pragma omp parallel for shared(result, left, right, rows, columns, inners) default(none) collapse(2)
  for (int rowTile = 0; rowTile < rows; rowTile += RowTileSize) {
    for (int columnTile = 0; columnTile < columns; columnTile += ColumnTileSize) {
      int rowTileEnd = std::min(rows, rowTile + RowTileSize);
      int columnTileEnd = std::min(columns, columnTile + ColumnTileSize);
      for (int innerTile = 0; innerTile < inners; innerTile += InnerTileSize) {
        int innerTileEnd = std::min(inners, innerTile + InnerTileSize);
        for (int row = rowTile; row < rowTileEnd; row++) {
          for (int inner = innerTile; inner < innerTileEnd; inner++) {
            for (int column = columnTile; column < columnTileEnd; column++) {
              result[row * columns + column] +=
                  left[row * inners + inner] * right[inner * columns + column];
            }
          }
        }
      }
    }
  }
}

inline void matmulImplSimdBasic(int rows, int columns, int inners,
                               const float *left, const float *right,
                               float *result) {
    std::fill_n(result, rows * columns, 0.0f);

    // Process one row at a time
    for (int row = 0; row < rows; row++) {
        // For each element in the inner dimension
        for (int inner = 0; inner < inners; inner++) {
            // Load the left matrix element and duplicate it to all elements
            // This creates a vector where all 4 elements are the same value
            float left_val = left[row * inners + inner];
            float32x4_t left_vec = vdupq_n_f32(left_val);

            // Process 4 columns at a time using NEON SIMD
            for (int col = 0; col < columns; col += 4) {
                if (col + 4 <= columns) {
                    // Load 4 elements from the right matrix
                    float32x4_t right_vec = vld1q_f32(&right[inner * columns + col]);

                    // Load current result values
                    float32x4_t result_vec = vld1q_f32(&result[row * columns + col]);

                    // Multiply and accumulate
                    // This is equivalent to:
                    // for (int i = 0; i < 4; i++)
                    //     result[i] += left_val * right[i];
                    result_vec = vfmaq_f32(result_vec, left_vec, right_vec);

                    // Store back the results
                    vst1q_f32(&result[row * columns + col], result_vec);
                } else {
                    // Handle remaining columns (less than 4) without SIMD
                    for (int remainder = col; remainder < columns; remainder++) {
                        result[row * columns + remainder] +=
                            left_val * right[inner * columns + remainder];
                    }
                }
            }
        }
    }
}

// Verify at compile time that all implementations satisfy the concept
static_assert(MatmulImplementation<decltype(matmulImplNaive)>);
static_assert(MatmulImplementation<decltype(matmulImplLoopOrder)>);
static_assert(MatmulImplementation<decltype(matmulImplAccelerate)>);
static_assert(MatmulImplementation<decltype(matmulImplTiling<3>)>);
static_assert(MatmulImplementation<decltype(matmulImplTilingRowCol<32, 32, 32>)>);
static_assert(MatmulImplementation<decltype(matmulImplSimdBasic)>);

