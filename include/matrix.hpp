#pragma once

#include <Accelerate/Accelerate.h>

// Concept
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

// Verify at compile time that all implementations satisfy the concept
static_assert(MatmulImplementation<decltype(matmulImplNaive)>);
static_assert(MatmulImplementation<decltype(matmulImplLoopOrder)>);
static_assert(MatmulImplementation<decltype(matmulImplAccelerate)>);
static_assert(MatmulImplementation<decltype(matmulImplTiling<3>)>);
