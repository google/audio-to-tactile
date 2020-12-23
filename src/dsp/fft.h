/* Copyright 2020 Google LLC
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
 * Complex-to-complex in-place fast Fourier transform (FFT) implementation for
 * transform sizes 4, 16, 64, 256, and 1024.
 *
 * The FFT algorithm is radix-4 Cooley-Tukey decimation in frequency for the
 * forward transform and radix-4 decimation in time for the inverse. This is
 * similar to the classic radix-2 algorithm
 * [http://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm], but
 * factoring into size-4 FFTs rather than size-2, which reduces the number of
 * multiplies by about 25%. For further arithmetic savings, the last radix-4
 * stage of the forward transform and first stage of the inverse transform are
 * implemented specially since they do not require twiddle factor
 * multiplications.
 *
 * The result of FftForwardScrambledTransform is the spectrum in bit-reversed
 * "scrambled" order. Conversely FftInverseScrambledTransform expects its input
 * in scrambled order. This allows the transforms to skip the overhead of
 * reordering, which for a size-256 transform reduces runtime by about 15%. Use
 * FftScramble and FftUnscramble to convert to and from scrambled order.
 *
 * Example uses:
 *
 * // Forward FFT example.
 * ComplexFloat data[256] = // Filled with waveform samples.
 * FftForwardScrambledTransform(data, 256);
 * FftUnscramble(data, 256);
 * // `data` array is now the spectrum of the waveform.
 *
 * // Inverse FFT example.
 * ComplexFloat data[256] = // Filled with a spectrum.
 * FftScramble(data, 256);
 * FftInverseScrambledTransform(data, 256);
 * // `data` array is now the inverse transform of the spectrum.
 *
 * // FFT-based convolution example. No scramble/unscramble steps needed.
 * ComplexFloat kernel[256] = // Filled with a convolution kernel.
 * ComplexFloat signal[256] = // Filled with a waveform samples.
 * FftForwardScrambledTransform(kernel, 256);
 * FftForwardScrambledTransform(signal, 256);
 * int k;
 * for (k = 0; k < 256; k++) { // Iterate frequencies in scrambled order.
 *   signal[k] = ComplexFloatMul(signal[k], kernel[k]);
 * }
 * FftInverseScrambledTransform(signal, 256);
 * // `signal` is now the circular convolution of the kernel with the waveform.
 *
 * Benchmarks (measured by extras/benchmark/fft_benchmark.cpp):
 * Results on SkyLake, 2020-11-06.
 * ----------------------------------------------------------------------------
 * Benchmark                                    Time           CPU   Iterations
 * ----------------------------------------------------------------------------
 * BM_FftForwardScrambledTransform/64         213 ns        213 ns     13068113
 * BM_FftForwardScrambledTransform/256       1169 ns       1169 ns      2394081
 * BM_FftForwardScrambledTransform/1024      5991 ns       5990 ns       469005
 * BM_FftInverseScrambledTransform/64         209 ns        209 ns     13354410
 * BM_FftInverseScrambledTransform/256       1149 ns       1149 ns      2439369
 * BM_FftInverseScrambledTransform/1024      5839 ns       5839 ns       479300
 * BM_FftScramble/64                         64.3 ns       64.3 ns     43634864
 * BM_FftScramble/256                         240 ns        240 ns     11714329
 * BM_FftScramble/1024                        957 ns        957 ns      2872617
 */

#ifndef AUDIO_TO_TACTILE_SRC_DSP_FFT_H_
#define AUDIO_TO_TACTILE_SRC_DSP_FFT_H_

#include "dsp/complex.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Performs in-place the forward complex-to-complex FFT, where the result is
 * scrambled in bit-reversed order. The transform size must be either 4, 16, 64
 * 256, or 1024. The data array is replaced with its spectrum:
 *
 *                N - 1
 *   data[R(k)] =  sum  data[n] exp(-i 2 pi k n / N),  for k = 0, ..., N - 1.
 *                n = 0
 *
 * where R(k) denotes the bit reversal of k. For example, if N = 64 and k = 11 =
 * 001011 (binary), then R(k) = 110100 (binary) = 52.
 *
 * Use `FftUnscramble(data, transform_size)` after this function to get the
 * spectrum in linear unscrambled order. After unscrambling, `data[k]` is the
 * spectral coefficient for frequency `k / transform_size` cycles per sample,
 * with `data[0]` corresponding to DC (constant) and `data[transform_size / 2]`
 * to Nyquist. If the original waveform was real-valued, then `data[k]` is the
 * complex conjugate of `data[transform_size - k]`.
 */
void FftForwardScrambledTransform(ComplexFloat* data, int transform_size);

/* Scrambles data in-place into bit-reversed order. */
void FftScramble(ComplexFloat* data, int transform_size);

/* Unscrambles data in-place. (An alias of FftScramble for readability.) */
static void FftUnscramble(ComplexFloat* data, int transform_size) {
  FftScramble(data, transform_size); /* FftScramble() is its own inverse. */
}

/* Performs in-place the (unnormalized) inverse complex-to-complex FFT, where
 * the input is scrambled in bit-reversed order. The transform size must be 4,
 * 16, 64, 256, or 1024. The data array is replaced with its inverse transform:
 *
 *             N - 1
 *   data[n] =  sum  data[R(k)] exp(+i 2 pi k n / N),  for n = 0, ..., N - 1.
 *             k = 0
 *
 * where R(k) denotes the bit reversal of k.
 *
 * Given a spectrum in usual linear order, rather than in scrambled order, use
 * `FftScramble(data, transform_size)` before this function to scramble it.
 *
 * This inverse transform is unnormalized: FftForwardTransform followed by
 * FftInverseTransform yields the original array scaled by transform_size.
 * To obtain the normalized inverse, each output element should be scaled by
 * 1 / transform_size. This scale factor can often be absorbed into computations
 * before or after the transforms, or simply ignored, to save a few multiplies.
 */
void FftInverseScrambledTransform(ComplexFloat* data, int transform_size);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* AUDIO_TO_TACTILE_SRC_DSP_FFT_H_ */
