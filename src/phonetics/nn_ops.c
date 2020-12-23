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

#include "phonetics/nn_ops.h"
#include "dsp/fast_fun.h"

static float DotProduct(const float* x, const float* y, int size) {
  float sum = 0.0f;
  int k;
  for (k = 0; k < size; ++k) {
    sum += x[k] * y[k];
  }
  return sum;
}

void DenseLinearLayer(int in_size,
                      int out_size,
                      const float* in,
                      const float* weights,
                      const float* bias,
                      float* out) {
  const float* weights_col_j = weights;
  int j;
  for (j = 0; j < out_size; ++j, weights_col_j += in_size) {
    out[j] = DotProduct(in, weights_col_j, in_size) + bias[j];
  }
}

static float Relu(float x) { return x > 0.0f ? x : 0.0f; }

void DenseReluLayer(int in_size,
                    int out_size,
                    const float* in,
                    const float* weights,
                    const float* bias,
                    float* out) {
  const float* weights_col_j = weights;
  int j;
  for (j = 0; j < out_size; ++j, weights_col_j += in_size) {
    out[j] = Relu(DotProduct(in, weights_col_j, in_size) + bias[j]);
  }
}

void Conv1DReluLayer(int in_frames,
                     int in_channels,
                     int out_channels,
                     int kernel_size,
                     const float* in,
                     const float* filters,
                     const float* bias,
                     float* out) {
  const int out_frames = in_frames - kernel_size + 1;
  const int dot_size = kernel_size * in_channels;
  int n;
  for (n = 0; n < out_frames; ++n) {
    /* Memory order is such that `sum_{dn, q} in[n + dn, q] * filters[q, dn, k]`
     * flattens to a contiguous 1D dot product,
     *
     *   sum_{dn, q} in[n + dn, q] * filters[q, dn, k]
     *   = sum_{dn, q} in[in_channels * (n + dn) + q]
     *               * filters[q + in_channels * (dn + kernel_size * k)]
     *   = sum_i in[in_channels * n + i] * filters[i + dot_size * k]
     *   = DotProduct(in + in_channels * n, filters_k, dot_size).
     */
    const float* filter_k = filters;
    int k;
    for (k = 0; k < out_channels; ++k, filter_k += dot_size) {
      out[k] = Relu(DotProduct(in, filter_k, dot_size) + bias[k]);
    }

    in += in_channels;
    out += out_channels;
  }
}

/* Max pool size is hard coded to 2. */
enum { kMaxPoolSize = 2 };

void MaxPool1DLayer(int in_frames,
                    int num_channels,
                    const float* in,
                    float* out) {
  const int out_frames = in_frames / kMaxPoolSize;  /* Integer division. */
  int n;
  for (n = 0; n < out_frames; ++n) {
    int c;
    for (c = 0; c < num_channels; ++c) {
      const float left = in[c];
      const float right = in[c + num_channels];
      out[c] = (left >= right) ? left : right;
    }
    in += kMaxPoolSize * num_channels;
    out += num_channels;
  }
}

void Softmax(float* x, int size) {
  float sum = 0.0f;
  int n;
  for (n = 0; n < size; ++n) {
    /* Clip x[n] to [-126, 126], since FastExp2() works only in this range. */
    const float temp = (x[n] < -126.0f) ? -126.0f : x[n];
    x[n] = FastExp2((temp > 126.0f) ? 126.0f : temp);
    sum += x[n];
  }

  const float scale = 1.0f / sum;
  for (n = 0; n < size; ++n) {
    x[n] *= scale;
  }
}
