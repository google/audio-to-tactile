/* Copyright 2020-2021 Google LLC
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

#include "dsp/q_resampler.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/q_resampler_kernel.h"

const QResamplerOptions kQResamplerDefaultOptions = {
    /*max_denominator=*/1000,
    /*rational_approximation_options=*/NULL,
    /*filter_radius_factor=*/5.0f,
    /*cutoff_proportion=*/0.9f,
    /*kaiser_beta=*/5.658f,
};

struct QResampler {
  /* Buffer of delayed input samples with capacity for num_taps frames. When
   * calling to ProcessSamples(), unconsumed input samples are stored in this
   * buffer so that they are available in the next call to ProcessSamples().
   */
  float* delayed_input;
  /* Polyphase filters, stored backward so that they can be applied as a dot
   * product. `filters[num_taps * p + k]` is the kth coefficient for phase p.
   */
  float* filters;
  /* Output buffer. Its capacity is large enough to hold the output from
   * resampling an input with size up to max(max_input_frames, FlushFrames).
   */
  float* output;
  /* Number of channels. */
  int num_channels;
  /* Number of frames currently in the `delayed_input` buffer. */
  int delayed_input_frames;
  /* Number of taps in each filter phase. */
  int num_taps;
  /* Radius of the filters in units of input samples. */
  int radius;
  /* Max supported number of input frames in calls to ProcessSamples(). */
  int max_input_frames;
  /* Max output size that ProcessSamples() can produce; capacity of `output`. */
  int max_output_frames;
  /* The rational approximating the requested resampling factor,
   *
   *   factor_numerator / factor_denominator
   *   ~= input_sample_rate_hz / output_sample_rate_hz,
   *
   * where factor_denominator is also the number of filter phases.
   */
  int factor_numerator;
  int factor_denominator;
  /* Equal to floor(factor_numerator / factor_denominator). */
  int factor_floor;
  /* Phase step between successive output samples, equal to
   * factor_numerator % factor_denominator.
   */
  int phase_step;
  int phase;
};

QResampler* QResamplerMake(float input_sample_rate_hz,
                           float output_sample_rate_hz,
                           int num_channels,
                           int max_input_frames,
                           const QResamplerOptions* options) {
  if (!options) {
    options = &kQResamplerDefaultOptions;
  }
  QResamplerKernel kernel;
  if (!QResamplerKernelInit(
          &kernel, input_sample_rate_hz, output_sample_rate_hz,
          /*filter_radius_factor=*/options->filter_radius_factor,
          /*cutoff_proportion=*/options->cutoff_proportion,
          /*kaiser_beta=*/options->kaiser_beta) ||
      num_channels <= 0 || max_input_frames <= 0 ||
      options->max_denominator <= 0) {
    return NULL;
  }

  const int radius = (int)ceil(kernel.radius);
  /* We create the polyphase filters h_p by sampling the kernel h(x) as
   *
   *   h_p[k] := h(p/b + k),  p = 0, 1, ..., b - 1,
   *
   * as described in the .h file. Since h(x) is nonzero for |x| <= radius,
   * h_p[k] is nonzero when |p/b + k| <= radius, or
   *
   *   -radius - p/b <= k <= radius - p/b.
   *
   * Independently of p, the nonzero support of h_p[k] is within
   *
   *   -radius - (b - 1)/b <= k <= radius.
   *
   * Since k and radius are integers, we can round the lower bound up to
   * conclude the nonzero support is within |k| <= radius. Therefore, we sample
   * h(p/b + k) for |k| <= radius, and the number of taps is 2 * radius + 1.
   */
  const int num_taps = 2 * radius + 1;
  /* Approximate resampling factor as a rational number, > 1 if downsampling. */
  int factor_numerator;
  int factor_denominator;
  RationalApproximation(kernel.factor, options->max_denominator,
                        options->rational_approximation_options,
                        &factor_numerator, &factor_denominator);
  /* For flushing, max_input_frames must be at least num_taps - 1. */
  if (num_taps - 1 > max_input_frames) {
    max_input_frames = num_taps - 1;
  }
  /* Get the max possible output size for the given max input size. */
  const int max_output_frames =
      (int)((((int64_t)max_input_frames) * factor_denominator +
             factor_numerator - 1) /
            factor_numerator);

  QResampler* resampler = (QResampler*)malloc(sizeof(QResampler));
  if (resampler == NULL) {
    return NULL;
  }

  resampler->filters = NULL;
  resampler->delayed_input = NULL;
  resampler->output = NULL;

  /* Allocate internal buffers. */
  if (!(resampler->filters =
            (float*)malloc(sizeof(float) * factor_denominator * num_taps)) ||
      !(resampler->delayed_input =
            (float*)malloc(sizeof(float) * num_taps * num_channels)) ||
      !(resampler->output =
            (float*)malloc(sizeof(float) * max_output_frames * num_channels))) {
    QResamplerFree(resampler);
    return NULL;
  }

  resampler->num_channels = num_channels;
  resampler->num_taps = num_taps;
  resampler->radius = radius;
  resampler->max_input_frames = max_input_frames;
  resampler->max_output_frames = max_output_frames;
  resampler->factor_numerator = factor_numerator;
  resampler->factor_denominator = factor_denominator;
  resampler->factor_floor =
      factor_numerator / factor_denominator; /* Integer divide. */
  resampler->phase_step = factor_numerator % factor_denominator;

  /* Compute polyphase resampling filter coefficients. */
  float* coeffs = resampler->filters;
  int phase;
  for (phase = 0; phase < factor_denominator; ++phase) {
    const double offset = ((double)phase) / factor_denominator;
    int k;
    for (k = -radius; k <= radius; ++k) {
      /* Store filter backwards so that convolution becomes a dot product. */
      coeffs[radius - k] = (float)QResamplerKernelEval(&kernel, offset + k);
    }
    coeffs += num_taps;
  }

  QResamplerReset(resampler);
  return resampler;
}

void QResamplerFree(QResampler* resampler) {
  if (resampler) {
    free(resampler->output);
    free(resampler->delayed_input);
    free(resampler->filters);
    free(resampler);
  }
}

void QResamplerReset(QResampler* resampler) {
  assert(resampler != NULL);
  int i;
  for (i = 0; i < resampler->radius * resampler->num_channels; ++i) {
    resampler->delayed_input[i] = 0.0f;
  }

  resampler->phase = 0;
  resampler->delayed_input_frames = resampler->radius;
}

void QResamplerGetRationalFactor(const QResampler* resampler,
                                 int* factor_numerator,
                                 int* factor_denominator) {
  assert(resampler != NULL);
  assert(factor_numerator != NULL);
  assert(factor_denominator != NULL);
  *factor_numerator = resampler->factor_numerator;
  *factor_denominator = resampler->factor_denominator;
}

float* QResamplerOutput(const QResampler* resampler) {
  return resampler->output;
}

int QResamplerNumChannels(const QResampler* resampler) {
  return resampler->num_channels;
}

int QResamplerMaxInputFrames(const QResampler* resampler) {
  return resampler->max_input_frames;
}

int QResamplerMaxOutputFrames(const QResampler* resampler) {
  return resampler->max_output_frames;
}

int QResamplerFlushFrames(const QResampler* resampler) {
  assert(resampler != NULL);
  /* ProcessSamples() continues until there are less than num_taps input frames.
   * By appending (num_taps - 1) zeros to the input, we guarantee that after the
   * call to ProcessSamples(), delayed_input is only zeros.
   *
   * NOTE: For API simplicity, this flush size is intentionally constant for a
   * resampler instance. It may be larger than necessary for the current state.
   * The flushed output has up to `num_taps / factor` more zeros than necessary.
   * For common parameters and sample rates, this is 10 to 30 samples and under
   * 2 ms, short enough that simplicity outweighs this minor inefficiency.
   */
  return resampler->num_taps - 1;
}

int QResamplerNextNumOutputFrames(const QResampler* resampler,
                                  int num_input_frames) {
  assert(resampler != NULL);
  assert(num_input_frames >= 0);
  const int min_consumed_input = 1 + num_input_frames +
                                 resampler->delayed_input_frames -
                                 resampler->num_taps;
  if (min_consumed_input <= 0) {
    return 0;
  }

  return (int)((((int64_t)min_consumed_input) * resampler->factor_denominator -
                resampler->phase + resampler->factor_numerator - 1) /
               resampler->factor_numerator);
}

int QResamplerProcessSamples(QResampler* resampler, const float* input,
                             int num_input_frames) {
  assert(resampler != NULL);
  assert(input != NULL);
  assert(num_input_frames >= 0);
  assert(resampler->delayed_input_frames < resampler->num_taps);
  assert(resampler->phase < resampler->factor_denominator);
  float* delayed_input = resampler->delayed_input;
  const int num_taps = resampler->num_taps;
  const int num_channels = resampler->num_channels;

  /* If num_input_frames is too big, drop some samples from the beginning. Drops
   * are of course always bad, no matter how they are handled. The user should
   * set `max_input_frames` large enough at construction to avoid drops.
   */
  const int excess_input = num_input_frames - resampler->max_input_frames;
  if (excess_input > 0) {
    /* Reset the resampler so that state before the drop does not influence
     * output produced after the drop.
     */
    QResamplerReset(resampler);
    input += excess_input * num_channels;
    num_input_frames -= excess_input;
  }

  if (resampler->delayed_input_frames + num_input_frames < num_taps) {
    if (num_input_frames > 0) {
      /* Append input to delayed_input. */
      memcpy(delayed_input + resampler->delayed_input_frames * num_channels,
             input, sizeof(float) * num_input_frames * num_channels);
      resampler->delayed_input_frames += num_input_frames;
    }
    return 0; /* Not enough frames available to produce any output yet. */
  }

  float* output = resampler->output;
  const int num_output_frames =
      QResamplerNextNumOutputFrames(resampler, num_input_frames);

  const float* filters = resampler->filters;
  const int factor_denominator = resampler->factor_denominator;
  const int factor_floor = resampler->factor_floor;
  const int phase_step = resampler->phase_step;
  int phase = resampler->phase;

  /* Below, the position in the input is (i + phase / factor_denominator) in
   * units of input samples, with `phase` tracking the fractional part.
   */
  int i = 0;
  /* `i` is the start index for applying the filters. To stay within the
   * available `delayed_input_frames + num_input_frames` input samples, we need
   *
   *   i + num_taps - 1 < delayed_input_frames + num_input_frames,
   *
   * or i < i_end = delayed_input_frames + num_input_frames - num_taps + 1.
   */
  int i_end = resampler->delayed_input_frames + num_input_frames - num_taps + 1;
  int num_written = 0;

  /* Process samples where the filter straddles delayed_input and input. */
  while (i < resampler->delayed_input_frames && i < i_end) {
    assert(num_written < num_output_frames);
    const int num_state = resampler->delayed_input_frames - i;
    const int num_input = num_taps - num_state;
    const float* filter = filters + phase * num_taps;

    int c;
    for (c = 0; c < num_channels; ++c) {
      float sum = 0.0f;

      int k;
      int offset = i * num_channels + c;
      /* Compute the dot product between `filter` and the concatenation of
       * `delayed_input[i:]` and `input[:num_input]`.
       */
      for (k = 0; k < num_state; ++k, offset += num_channels) {
        sum += filter[k] * delayed_input[offset];
      }
      for (k = 0, offset = c; k < num_input; ++k, offset += num_channels) {
        sum += filter[num_state + k] * input[offset];
      }

      output[c] = sum;
    }

    output += num_channels;
    ++num_written;
    i += factor_floor;
    phase += phase_step;
    if (phase >= factor_denominator) {
      phase -= factor_denominator;
      ++i;
    }
  }

  if (i < resampler->delayed_input_frames) {
    /* Ran out of input samples before consuming everything in delayed_input.
     * Discard the samples of delayed_input that have been consumed and append
     * the input.
     */
    assert(num_written == num_output_frames);
    int remaining = resampler->delayed_input_frames - i;
    memmove(delayed_input, delayed_input + num_channels * i,
            sizeof(float) * remaining * num_channels);
    memcpy(delayed_input + remaining * num_channels, input,
           sizeof(float) * num_input_frames * num_channels);
    resampler->delayed_input_frames += num_input_frames - i;
    assert(resampler->delayed_input_frames < resampler->num_taps);
    resampler->phase = phase;
    return num_output_frames;
  }

  /* Consumed everything in delayed_input_. Now process output samples that
   * depend on only the input.
   */
  i -= resampler->delayed_input_frames;
  i_end -= resampler->delayed_input_frames;

  if (num_channels == 1) {
    /* Specialization for num_channels == 1. Improves benchmark by 10%. */
    int count = 0;
    while (i < i_end) {
      assert(num_written < num_output_frames);
      const float* filter = filters + phase * num_taps;

      float sum = 0.0f;
      int k;
      for (k = 0; k < num_taps; ++k) {
        sum += filter[k] * input[i + k];
      }
      output[count] = sum;
      ++count;

      i += factor_floor;
      phase += phase_step;
      if (phase >= factor_denominator) {
        phase -= factor_denominator;
        ++i;
      }
    }
    num_written += count;
  } else { /* General implementation for arbitrary num_channels. */
    while (i < i_end) {
      assert(num_written < num_output_frames);
      const float* filter = filters + phase * num_taps;

      int c;
      for (c = 0; c < num_channels; ++c) {
        float sum = 0.0f;

        int k;
        int offset = i * num_channels + c;
        for (k = 0; k < num_taps; ++k, offset += num_channels) {
          sum += filter[k] * input[offset];
        }

        output[c] = sum;
      }
      output += num_channels;

      ++num_written;
      i += factor_floor;
      phase += phase_step;
      if (phase >= factor_denominator) {
        phase -= factor_denominator;
        ++i;
      }
    }
  }

  assert(num_written == num_output_frames);
  assert(i <= num_input_frames);
  resampler->delayed_input_frames = num_input_frames - i;
  memcpy(delayed_input, input + i * num_channels,
         sizeof(float) * resampler->delayed_input_frames * num_channels);
  resampler->phase = phase;
  return num_output_frames;
}
