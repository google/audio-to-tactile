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
 *
 *
 * Ops for neural net inference.
 */

#ifndef AUDIO_TO_TACTILE_TACTILE_NN_OPS_H_
#define AUDIO_TO_TACTILE_TACTILE_NN_OPS_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Computes in * weights + bias to perform dense (fully-connected) layer,
 *
 *   out[j] = (sum_k in[k] * weights[k, j]) + bias[j],
 *
 * where
 *   `in` is an input array of size in_size,
 *   `weights` is a column-major matrix of shape [in_size, out_size],
 *   `bias` is an array of size out_size,
 *   `out` is an output array of size out_size.
 */
void DenseLinearLayer(int in_size,
                      int out_size,
                      const float* in,
                      const float* weights,
                      const float* bias,
                      float* out);

/* Same as above but with ReLU activation,
 *
 *   out[j] = relu((sum_k in[k] * weights[k, j]) + bias[j]).
 */
void DenseReluLayer(int in_size,
                    int out_size,
                    const float* in,
                    const float* weights,
                    const float* bias,
                    float* out);

/* Computes a 1D conv layer with ReLU activation,
 *
 *   out[n, k] = relu(sum_{dn, q} in[n + dn, q] * filters[q, dn, k] + bias[k]).
 *
 * where
 *   `in` is a row-major matrix of shape [in_frames, in_channels],
 *   `filters` is a column-major 3D tensor of shape
 *      [in_channels, kernel_size, out_channels],
 *   `bias` is an array of size out_channels,
 *   `out` is a row-major matrix of shape
 *      [in_frames + kernel_size - 1, out_channels].
 */
void Conv1DReluLayer(int in_frames,
                     int in_channels,
                     int out_channels,
                     int kernel_size,
                     const float* in,
                     const float* filters,
                     const float* bias,
                     float* out);

/* Computes 1D max pooling with a pool size (decimation factor) of 2,
 *
 *   out[n, c] = max(in[2 * n, c], in[2 * n + 1, c]).
 *
 * where
 *   `in` is a row-major matrix of shape [in_frames, num_channels],
 *   `out` is a row-major matrix of shape [floor(in_frames / 2), num_channels].
 *
 * NOTE: For speed, the pool size is hard coded to 2.
 */
void MaxPool1DLayer(int in_frames,
                    int num_channels,
                    const float* in,
                    float* out);

/* Softmax with a base of 2, computed in place,
 *
 *   out[n] = 2^in[n] / (2^in[0] + 2^in[1] + ... + 2^in[size - 1]).
 *
 * Powers of 2 are approximately evaluated with FastExp2().
 */
void Softmax(float* x, int size);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_TACTILE_NN_OPS_H_ */

