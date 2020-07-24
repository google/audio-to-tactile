/* Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "audio/tactile/nn_ops.h"

#include <math.h>
#include <stdlib.h>

#include "audio/dsp/portable/logging.h"

void TestDenseLayers() {
  float* in = (float*) CHECK_NOTNULL(malloc(3 * sizeof(float)));
  float* out = (float*) CHECK_NOTNULL(malloc(2 * sizeof(float)));

  /* 3x2 weights matrix (stored in column-major order):
   *   [ 1.0  4.0]
   *   [ 2.0  5.0]
   *   [-3.0  6.0]
   */
  static const float kWeights[3 * 2] = {1.0f, 2.0f, -3.0f, 4.0f, 5.0f, 6.0f};
  static const float kBias[2] = {0.1f, 0.2f};

  in[0] = 1.0f;
  in[1] = 0.0f;
  in[2] = -2.0f;
  DenseLinearLayer(3, 2, in, kWeights, kBias, out);
  CHECK(fabs(out[0] - 7.1f) <= 1e-6f);
  CHECK(fabs(out[1] - -7.8f) <= 1e-6f);
  DenseReluLayer(3, 2, in, kWeights, kBias, out);
  CHECK(fabs(out[0] - 7.1f) <= 1e-6f);
  CHECK(fabs(out[1] - 0.0f) <= 1e-6f);

  in[0] = -2.0f;
  in[1] = 1.5f;
  in[2] = 2.0f;
  DenseLinearLayer(3, 2, in, kWeights, kBias, out);
  CHECK(fabs(out[0] - -4.9f) <= 1e-6f);
  CHECK(fabs(out[1] - 11.7f) <= 1e-6f);
  DenseReluLayer(3, 2, in, kWeights, kBias, out);
  CHECK(fabs(out[0] - 0.0f) <= 1e-6f);
  CHECK(fabs(out[1] - 11.7f) <= 1e-6f);

  free(out);
  free(in);
}

static void FillRandomValues(float* x, int size) {
  int i;
  for (i = 0; i < size; ++i) {
    x[i] = rand() / (float)RAND_MAX - 0.5f;
  }
}

void TestConv1DReluLayer(int in_channels, int out_channels) {
  const int kInFrames = 5;
  const int kKernelSize = 3;
  const int kOutFrames = kInFrames - kKernelSize + 1;

  float* in = (float*) CHECK_NOTNULL(malloc(
      kInFrames * in_channels * sizeof(float)));
  float* filters = (float*) CHECK_NOTNULL(malloc(
      out_channels * kKernelSize * in_channels * sizeof(float)));
  float* bias = (float*) CHECK_NOTNULL(malloc(
      out_channels * sizeof(float)));
  float* out = (float*) CHECK_NOTNULL(malloc(
      kOutFrames * out_channels * sizeof(float)));

  FillRandomValues(in, kInFrames * in_channels);
  FillRandomValues(filters, out_channels * kKernelSize * in_channels);
  FillRandomValues(bias, out_channels);
  FillRandomValues(out, kOutFrames * out_channels);

  Conv1DReluLayer(kInFrames, in_channels, out_channels, kKernelSize,
                  in, filters, bias, out);

  int n;
  for (n = 0; n < kOutFrames; ++n) {
    int k;
    for (k = 0; k < out_channels; ++k) {
      /* Compute the (n, k)th expected output element,
       *   relu(sum_{dn, q} in[n + dn, q] * filters[q, dn, k] + bias[k]).
       */
      float sum = bias[k];
      int dn;
      for (dn = 0; dn < kKernelSize; ++dn) {
        int q;
        for (q = 0; q < in_channels; ++q) {
          sum += in[in_channels * (n + dn) + q]
              * filters[q + in_channels * (dn + kKernelSize * k)];
        }
      }

      if (sum < 0.0f) { sum = 0.0f; }  /* Apply ReLU. */

      CHECK(fabs(sum - out[out_channels * n + k]) < 1e-6f);
    }
  }

  free(out);
  free(bias);
  free(filters);
  free(in);
}

void TestMaxPool1DLayer() {
  /* Input with 7 frames and 2 channels. */
  static const float in[7 * 2] = {-2.1f, 3.9f,
                                  0.3f, 1.8f,
                                  -4.2f, -3.3f,
                                  -0.9f, -0.3f,
                                  2.4f, 0.3f,
                                  -3.9f, 1.5f,
                                  -0.9f, 2.1f};
  float* out = (float*) CHECK_NOTNULL(malloc(3 * 2 * sizeof(float)));
  MaxPool1DLayer(7, 2, in, out);

  /* Expected output with 3 frames and 2 channels. */
  static const float kExpected[3 * 2] = {0.3f, 3.9f,
                                         -0.9f, -0.3f,
                                         2.4f, 1.5f};
  int i;
  for (i = 0; i < 3 * 2; ++i) {
    CHECK(fabs(out[i] - kExpected[i]) < 1e-6f);
  }

  free(out);
}

int main(int argc, char** argv) {
  srand(0);
  TestDenseLayers();
  TestConv1DReluLayer(1, 1);
  TestConv1DReluLayer(3, 2);
  TestConv1DReluLayer(2, 3);
  TestMaxPool1DLayer();

  puts("PASS");
  return EXIT_SUCCESS;
}

