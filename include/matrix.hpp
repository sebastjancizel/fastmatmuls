#pragma once

#include <Accelerate/Accelerate.h>
#include <vecLib/cblas.h>

template <int Rows, int Columns, int Inners>
void matmulImplNaive(const float *left, const float *right, float *result) {
  for (int row = 0; row < Rows; row++) {
    for (int column = 0; column < Columns; column++) {
      float sum = 0.0f;
      for (int inner = 0; inner < Inners; inner++) {
        sum += left[row * Inners + inner] * right[inner * Columns + column];
      }
      result[row * Columns + column] = sum;
    }
  }
}

template <int Rows, int Columns, int Inners>
void matmulImplLoopOrder(const float *left, const float *right, float *result) {
  for (int row = 0; row < Rows; row++) {
    for (int inner = 0; inner < Inners; inner++) {
      for (int column = 0; column < Columns; column++) {
        result[row * Columns + column] +=
            left[row * Inners + inner] * right[inner * Columns + column];
      }
    }
  }
}

template <int Rows, int Columns, int Inners>
void matmulImplAccelerate(const float *left, const float *right,
                          float *result) {

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Rows, Columns, Inners,
              1.0f, left, Inners, right, Columns, 0.0f, result, Columns);
}
