#pragma once

#include <Accelerate/Accelerate.h>
#include <vecLib/cblas.h>

void matmulImplNaive(int rows, int columns, int inners, const float *left,
                     const float *right, float *result) {
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

void matmulImplLoopOrder(int rows, int columns, int inners, const float *left,
                         const float *right, float *result) {
  for (int row = 0; row < rows; row++) {
    for (int inner = 0; inner < inners; inner++) {
      for (int column = 0; column < columns; column++) {
        result[row * columns + column] +=
            left[row * inners + inner] * right[inner * columns + column];
      }
    }
  }
}

void matmulImplAccelerate(int rows, int columns, int inners, const float *left,
                          const float *right, float *result) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, columns, inners,
              1.0f, left, inners, right, columns, 0.0f, result, columns);
}
