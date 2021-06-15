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
 *
 *
 * Computes a smoothed signal envelope.
 *
 * This library (1) applies a highpass filter, (2) computes a smoothed signal
 * power envelope, (3) samples measurements of the envelope at regularly-spaced
 * points, and (4) serializes batches of these measurements in a "record". The
 * record uses an efficient delta encoding to compress to a BLE-friendly size. A
 * batch of 33 measurements is encoded in 13 bytes.
 *
 * Example use:
 *   EnvelopeTracker tracker;
 *   EnvelopeTrackerInit(&tracker, 16000);
 *
 *   while (...) {
 *     const float* input = // Get next input samples.
 *     if (EnvelopeTrackerProcessSamples(&tracker, input, num_samples)) {
 *       uint8_t record[kEnvelopeTrackerRecordBytes];
 *       EnvelopeTrackerGetRecord(&tracker, record);
 *       // Save the record.
 *     }
 *   }
 *
 *   // Decoding a record.
 *   uint8_t record[kEnvelopeTrackerRecordBytes] = ...
 *   float powers[kEnvelopeTrackerRecordPoints];
 *   EnvelopeTrackerDecodeRecord(record, powers);
 */

#ifndef AUDIO_TO_TACTILE_SRC_TACTILE_ENVELOPE_TRACKER_H_
#define AUDIO_TO_TACTILE_SRC_TACTILE_ENVELOPE_TRACKER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Number of measurement points per record. Must be a multiple of 8 plus one. */
#define kEnvelopeTrackerRecordPoints 33
/* Bytes in a record: 1 byte for first point + 3 bits per subsequent point. */
#define kEnvelopeTrackerRecordBytes \
  (1 + (kEnvelopeTrackerRecordPoints - 1) * 3 / 8)

typedef struct {
  /* Smoother coeff for updating the low-frequency signal trend. */
  float trend_smoother;
  /* Current signal trend, which is subtracted to make the highpass filter. */
  float trend;
  /* Smoother coeff for smoothing the energy envelope. */
  float energy_smoother;
  /* The current energy. */
  float energy;
  /* Number of input samples between successive measurements. */
  int samples_per_measurement;
  /* Counter counting down the number of samples until the next measurement. */
  int sample_count;
  /* Number of measurements in `buffer`. */
  int num_measurements;
  /* Buffer for measurements, also used to hold the serialized record. */
  uint8_t buffer[kEnvelopeTrackerRecordPoints];
} EnvelopeTracker;

/* Initializes the tracker with the specified sample rate. */
void EnvelopeTrackerInit(EnvelopeTracker* tracker, float sample_rate_hz);

/* Processes `num_samples` input samples in a streaming manner. It is assumed
 * that `num_samples` is strictly less than 60 ms worth of input (specifically,
 * `2 * tracker->samples_per_measurement`), otherwise record data will be
 * overwritten before it could be read. The function returns 1 if the next
 * record is ready and 0 otherwise.
 */
int /*bool*/ EnvelopeTrackerProcessSamples(EnvelopeTracker* tracker,
                                           const float* input,
                                           int num_samples);

/* Copies the encoded record to `dest`, which must have space for
 * kEnvelopeTrackerRecordBytes bytes. Call this function when
 * EnvelopeTrackerProcessSamples returns 1.
 */
void EnvelopeTrackerGetRecord(const EnvelopeTracker* tracker, uint8_t* dest);

/* Decodes a record as an array of power values. `dest` must have space for
 * kEnvelopeTrackerRecordPoints elements.
 */
void EnvelopeTrackerDecodeRecord(const uint8_t* record, float* dest);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_SRC_TACTILE_ENVELOPE_TRACKER_H_ */
