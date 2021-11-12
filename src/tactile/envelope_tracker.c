/* Copyright 2021 Google LLC
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

#include "tactile/envelope_tracker.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/fast_fun.h"
#include "dsp/math_constants.h"

/* Cutoff for the highpass filter, which is applied before computing energy. */
#define kHighpassCutoffHz 20.0f
/* Time constant in seconds for smoothing the energy envelope. */
#define kEnergyTimeConstant 0.15f
/* Period in seconds between successive measurements. */
#define kMeasurementPeriod 0.03f

/* Encodes `energy` as a value between 0 and 255.
 * In order to represent a wide range of energies, a power law x^1/6 is applied
 * before quantization. An energy of zero is represented exactly, and the
 * positive representable energy range is about 4e-15 to 1.0 (corresponding
 * to amplitudes in 6e-8 to 1.0). The representable energies are close to being
 * geometrically spaced with a spacing of 12% for energies around 10^-4 (in
 * terms of amplitudes, 6% around 10^-2), and spaced a little tighter the larger
 * the energy. So the representation is unlikely to over or under flow and is
 * quantized finely enough to approximate the energy within a few percent.
 */
static uint8_t EncodeEnergy(float energy) {
  float value = 255.0f * FastPow(energy, 1.0f / 6);
  if (value < 0.0f) { value = 0.0f; }
  if (value > 255.0f) { value = 255.0f; }
  return (uint8_t)(value + 0.5f);
}
/* Decodes an energy value, the reverse of EncodeEnergy(). */
static float DecodeEnergy(uint8_t value) {
  return FastPow((float)value / 255.0f, 6.0f);
}

/* Encodes an integer delta as a 3-bit code between 0 and 7.
 * The correspondence between codes and deltas is:
 *
 *   code  delta      code  delta
 *      0      0         4      0 (redundant with code 0)
 *      1     +1         5     -1
 *      2     +4         6     -4
 *      3    +11         7    -11
 *
 * This function is used to encode the delta between two successive envelope
 * measurements, after having already encoded each of them with EncodeEnergy().
 * Since the envelope is smoothed, it is expected that nearby measurements are
 * similar, so the values encoded by EncodeEnergy() are often the same or close.
 *
 * The idea with the above correspondence is that codes 3 and 7 approximate
 * quick changes, |delta| >= 8, while the other codes are for slower changes and
 * fine corrections to previous deltas. For instance a change of +3 is
 * represented by +4 (code 2) followed by -1 (code 5). For simplicity and
 * symmetry, codes 4 and 0 both represent a delta of zero.
 */
static uint8_t EncodeDelta(int delta) {
  uint8_t code;
  int magnitude = abs(delta);
  if (magnitude >= 8) {
    code = 3;
  } else {
    /* Map magnitude to code.  index: 0, 1, 2, 3, 4, 5, 6, 7. */
    static const uint8_t kCodes[8] = {0, 1, 1, 2, 2, 2, 2, 2};
    code = kCodes[magnitude];
  }

  if (delta < 0) { code |= 4; }
  return code;
}
/* Decodes a delta, the reverse of EncodeDelta(). */
static int DecodeDelta(int code) {
  /* Map code to delta.      index: 0, 1, 2,  3, 4,  5,  6,   7. */
  static const int8_t kDeltas[8] = {0, 1, 4, 11, 0, -1, -4, -11};
  return kDeltas[code];
}

/* Encodes the record in-place in `buffer`. */
static void EncodeRecord(EnvelopeTracker* tracker) {
  const uint8_t* src = tracker->buffer;
  uint8_t* record = tracker->buffer;
  /* The initial encoded energy is already in the first byte of the buffer. */
  int cumulative = src[0];
  int i;
  /* Each iteration in i handles 8 measurements, encoding them as 24 bits. */
  for (i = 1; i < kEnvelopeTrackerRecordPoints; i += 8, record += 3) {
    uint_fast32_t pack24 = 0;
    int j;
    for (j = 0; j < 8; ++j) {
      /* Encode the delta between the current measurement and `cumulative`. */
      const uint_fast32_t code = EncodeDelta((int)src[i + j] - cumulative);
      /* To account for error, decode the delta and add it to `cumulative`. */
      cumulative += DecodeDelta(code);
      /* Store the 3-bit code in little endian order. */
      pack24 |= code << (3 * j);
    }

    record[1] = (uint8_t)(pack24);
    record[2] = (uint8_t)(pack24 >> 8);
    record[3] = (uint8_t)(pack24 >> 16);
  }
}

void EnvelopeTrackerDecodeRecord(const uint8_t* record, float* dest) {
  int cumulative = record[0]; /* Initial energy is in the first byte. */
  dest[0] = DecodeEnergy(cumulative);

  int i;
  for (i = 1; i < kEnvelopeTrackerRecordPoints; i += 8, record += 3) {
    uint_fast32_t pack24 = (uint_fast32_t)(record[1])
                         | (uint_fast32_t)(record[2]) << 8
                         | (uint_fast32_t)(record[3]) << 16;
    int j;
    for (j = 0; j < 8; ++j, pack24 >>= 3) {
      /* Get the next 3-bit delta code, decode it, and add to `cumulative`. */
      cumulative += DecodeDelta(pack24 & 7);
      dest[i + j] = DecodeEnergy(cumulative);
    }
  }
}

/* Records `energy` as the next measurement, saving it in `buffer`. If the
 * buffer is filled, the record is encoded and the function returns 1.
 * Otherwise, it returns 0.
 */
static int /*bool*/ RecordMeasurement(EnvelopeTracker* tracker, float energy) {
  tracker->buffer[tracker->num_measurements] = EncodeEnergy(energy);
  tracker->num_measurements++;

  if (tracker->num_measurements == kEnvelopeTrackerRecordPoints) {
    EncodeRecord(tracker);
    tracker->num_measurements = 0;
    return 1;
  }
  return 0;
}

void EnvelopeTrackerInit(EnvelopeTracker* tracker, float sample_rate_hz) {
  /* Compute smoother coeffs for one-pole filters. */
  tracker->trend_smoother =
      (float)(1 - exp(-2 * M_PI * kHighpassCutoffHz / sample_rate_hz));
  tracker->energy_smoother =
      (float)(1 - exp(-1 / (kEnergyTimeConstant * sample_rate_hz)));
  /* Convert kMeasurementPeriod from seconds to samples. */
  tracker->samples_per_measurement =
      (int)(kMeasurementPeriod * sample_rate_hz + 0.5f);

  /* Set zero initial state. */
  tracker->trend = 0.0f;
  tracker->energy = 0.0f;
  tracker->sample_count = tracker->samples_per_measurement;
  tracker->num_measurements = 0;
}

int /*bool*/ EnvelopeTrackerProcessSamples(EnvelopeTracker* tracker,
                                           const float* input,
                                           int num_samples) {
  float trend = tracker->trend;
  float energy = tracker->energy;
  int sample_count = tracker->sample_count;
  int record_ready = 0;

  while (num_samples > 0) {
    /* Determine the number of samples until the next measurement or input runs
     * out, whichever comes first.
     */
    const int num_iter =
        (num_samples < sample_count) ? num_samples : sample_count;

    int i;
    for (i = 0; i < num_iter; ++i) {
      /* Perform one-pole highpass filter. */
      float sample = input[i];
      trend += tracker->trend_smoother * (sample - trend);
      sample -= trend;
      /* Square to compute energy and smooth with one-pole filter. */
      energy += tracker->energy_smoother * (sample * sample - energy);
    }

    if (!(sample_count -= num_iter)) {
      /* Record the next measurement. */
      record_ready |= RecordMeasurement(tracker, energy);
      input += num_iter;
      num_samples -= num_iter;
      sample_count = tracker->samples_per_measurement;
    } else {
      break;
    }
  }

  tracker->trend = trend;
  tracker->energy = energy;
  tracker->sample_count = sample_count;
  return record_ready;
}

void EnvelopeTrackerGetRecord(const EnvelopeTracker* tracker, uint8_t* dest) {
  memcpy(dest, tracker->buffer, kEnvelopeTrackerRecordBytes);
}
