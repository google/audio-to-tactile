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

#include "extras/references/taps/tactile_player.h"

#include <math.h>
#include <pthread.h>
#include <string.h>

#include "src/dsp/logging.h"

#ifndef M_1_SQRT2
#define M_1_SQRT2 0.70710678118654752440
#endif /* M_1_SQRT2 */

void TestStreaming() {
  const int kNumChannels = 4;
  const float kSampleRateHz = 44100.0f;
  const int kBufferFrames = 5;
  const int kTotalFrames = kBufferFrames + 2;

  float* buffer = (float*)CHECK_NOTNULL(
      malloc(kBufferFrames * kNumChannels * sizeof(float)));
  TactilePlayer* player =
      CHECK_NOTNULL(TactilePlayerMake(kNumChannels, kSampleRateHz));
  CHECK(!TactilePlayerIsActive(player));

  float* samples = (float*)CHECK_NOTNULL(
      malloc(kTotalFrames * kNumChannels * sizeof(float)));
  int i;
  int c;
  for (i = 0; i < kTotalFrames; ++i) {
    for (c = 0; c < kNumChannels; ++c) {
      samples[kNumChannels * i + c] = i + 0.1f * c;
    }
  }

  /* Play signal with `kBufferFrames + 2` frames. */
  TactilePlayerPlay(player, samples, kTotalFrames);
  CHECK(TactilePlayerIsActive(player));

  /* First time filling buffer produces kBufferFrames playback frames. */
  CHECK(TactilePlayerFillBuffer(player, kBufferFrames, buffer) ==
        kBufferFrames);
  CHECK(memcmp(buffer, samples, kBufferFrames * kNumChannels * sizeof(float)) ==
        0);

  /* Second time filling buffer produces the 2 remaining playback frames. */
  CHECK(TactilePlayerFillBuffer(player, kBufferFrames, buffer) == 2);
  CHECK(memcmp(buffer, samples + kBufferFrames * kNumChannels,
               2 * kNumChannels * sizeof(float)) == 0);
  /* Other frames in the buffer have been zero filled. */
  for (i = 2; i < kBufferFrames; ++i) {
    for (c = 0; c < kNumChannels; ++c) {
      CHECK(buffer[kNumChannels * i + c] == 0.0f);
    }
  }

  /* Third time filling buffer produces no playback frames. */
  CHECK(TactilePlayerFillBuffer(player, kBufferFrames, buffer) == 0);
  /* Buffer should be zero filled. */
  for (i = 0; i < kBufferFrames; ++i) {
    for (c = 0; c < kNumChannels; ++c) {
      CHECK(buffer[kNumChannels * i + c] == 0.0f);
    }
  }

  TactilePlayerFree(player);
  free(buffer);
}

static float SignalA(int c, float t) { return sin((15 + c) * 1000 * t); }
static float SignalB(int c, float t) { return sin((5 + c) * 1000 * t); }

void TestInterruptedPlayback() {
  const int kNumChannels = 4;
  const float kSampleRateHz = 8000.0f;
  const int kBufferFrames = 64;
  const int kTotalFrames = 140;
  const float kFadeOutDuration = 0.005f;

  float* buffer = (float*)CHECK_NOTNULL(
      malloc(kBufferFrames * kNumChannels * sizeof(float)));
  TactilePlayer* player =
      CHECK_NOTNULL(TactilePlayerMake(kNumChannels, kSampleRateHz));

  float* samples_a = (float*)CHECK_NOTNULL(
      malloc(kTotalFrames * kNumChannels * sizeof(float)));
  float* samples_b = (float*)CHECK_NOTNULL(
      malloc(kTotalFrames * kNumChannels * sizeof(float)));
  int i;
  int c;
  for (i = 0; i < kTotalFrames; ++i) {
    const float t = i / kSampleRateHz;
    for (c = 0; c < kNumChannels; ++c) {
      samples_a[kNumChannels * i + c] = SignalA(c, t);
      samples_b[kNumChannels * i + c] = SignalB(c, t);
    }
  }

  /* Start signal A and play the first kBufferFrames frames. */
  TactilePlayerPlay(player, samples_a, kTotalFrames);
  CHECK(TactilePlayerFillBuffer(player, kBufferFrames, buffer) ==
        kBufferFrames);
  CHECK(memcmp(buffer, samples_a,
               kBufferFrames * kNumChannels * sizeof(float)) == 0);

  /* Interrupt signal A playback with signal B. */
  TactilePlayerPlay(player, samples_b, kTotalFrames);
  CHECK(TactilePlayerFillBuffer(player, kBufferFrames, buffer) ==
        kBufferFrames);

  const float offset_a = kBufferFrames / kSampleRateHz;
  for (i = 0; i < kBufferFrames; ++i) {
    const float t = i / kSampleRateHz;
    float fade_weight = 1.0f - (t + 0.5f / kSampleRateHz) / kFadeOutDuration;
    if (fade_weight < 0.0f) {
      fade_weight = 0.0f;
    }

    for (c = 0; c < kNumChannels; ++c) {
      const float expected =
          SignalB(c, t) + fade_weight * SignalA(c, t + offset_a);
      CHECK(fabs(buffer[kNumChannels * i + c] - expected) <= 2e-5f);
    }
  }

  TactilePlayerFree(player);
  free(buffer);
}

float EnvelopeChannel0(float t) { return 30 * t; }
float EnvelopeChannel1(float t) { return 1.3 - cos(400 * t); }

void TestGetRms() {
  const int kNumChannels = 2;
  const float kSampleRateHz = 16000.0f;
  const int kBufferFrames = 32;
  const int kTotalFrames = 600;

  float* buffer = (float*)CHECK_NOTNULL(
      malloc(kBufferFrames * kNumChannels * sizeof(float)));
  TactilePlayer* player =
      CHECK_NOTNULL(TactilePlayerMake(kNumChannels, kSampleRateHz));

  float* samples = (float*)CHECK_NOTNULL(
      malloc(kTotalFrames * kNumChannels * sizeof(float)));
  int i;
  for (i = 0; i < kTotalFrames; ++i) {
    const float t = i / kSampleRateHz;
    const float carrier = sin(5000 * t);
    samples[kNumChannels * i + 0] = carrier * EnvelopeChannel0(t);
    samples[kNumChannels * i + 1] = carrier * EnvelopeChannel1(t);
  }

  TactilePlayerPlay(player, samples, kTotalFrames);

  for (i = kBufferFrames; i < kTotalFrames - kBufferFrames;
       i += kBufferFrames) {
    CHECK(TactilePlayerFillBuffer(player, kBufferFrames, buffer) ==
          kBufferFrames);

    /* Get RMS at the current time, estimated over a 1ms window. */
    const float t = i / kSampleRateHz;
    float rms[2];
    TactilePlayerGetRms(player, 0.001f, rms);

    /* RMS in each channel should be close to 1/sqrt(2) * envelope. */
    CHECK(fabs(rms[0] - M_1_SQRT2 * EnvelopeChannel0(t)) <= 0.07);
    CHECK(fabs(rms[1] - M_1_SQRT2 * EnvelopeChannel1(t)) <= 0.07);
  }

  TactilePlayerFree(player);
  free(buffer);
}

struct MockPlaybackInfo {
  TactilePlayer* player;
  pthread_mutex_t lock;
  int keep_running; /* Guarded by `lock`. */
  float* tactor_output;
  int num_tactor_output;
};
typedef struct MockPlaybackInfo MockPlaybackInfo;

void* MockPlayback(void* arg) {
  MockPlaybackInfo* info = (MockPlaybackInfo*)arg;
  float buffer[4];
  int keep_running = 1;

  do {
    int num_frames = TactilePlayerFillBuffer(info->player, 4, buffer);
    /* A real application would give `buffer` to portaudio or similar to play
     * on a speaker. We mock this out by appending to `tactor_output`.
     */
    memcpy(info->tactor_output + info->num_tactor_output, buffer,
           num_frames * sizeof(float));
    info->num_tactor_output += num_frames;

    pthread_mutex_lock(&info->lock);
    keep_running = info->keep_running;
    pthread_mutex_unlock(&info->lock);
  } while (keep_running);

  return NULL;
}

void TestPlaybackThread() {
  const int kNumChannels = 1;
  const float kSampleRateHz = 8000.0f;

  MockPlaybackInfo info;
  info.player = CHECK_NOTNULL(TactilePlayerMake(kNumChannels, kSampleRateHz));
  CHECK(pthread_mutex_init(&info.lock, NULL) == 0);
  info.keep_running = 1;
  info.tactor_output = (float*)CHECK_NOTNULL(malloc(16 * sizeof(float)));
  info.num_tactor_output = 0;

  /* Create a thread that performs a mock version of playback. It runs
   * TactilePlayerFillBuffer in a loop and "play" output by appending to
   * `tactor_output` buffer.
   */
  pthread_t playback_thread;
  CHECK(pthread_create(&playback_thread, NULL, MockPlayback, &info) == 0);

  /* Play a 5-sample signal. */
  float* samples = (float*)CHECK_NOTNULL(malloc(5 * sizeof(float)));
  samples[0] = 10.0f;
  samples[1] = 20.0f;
  samples[2] = 30.0f;
  samples[3] = 40.0f;
  samples[4] = 50.0f;
  TactilePlayerPlay(info.player, samples, 5);

  while (TactilePlayerIsActive(info.player)) {
  } /* Wait while playing. */

  /* Play a 3-sample signal. */
  samples = (float*)CHECK_NOTNULL(malloc(3 * sizeof(float)));
  samples[0] = 60.0f;
  samples[1] = 70.0f;
  samples[2] = 80.0f;
  TactilePlayerPlay(info.player, samples, 3);

  while (TactilePlayerIsActive(info.player)) {
  } /* Wait while playing. */

  pthread_mutex_lock(&info.lock);
  info.keep_running = 0; /* Stop the playback thread. */
  pthread_mutex_unlock(&info.lock);
  CHECK(pthread_join(playback_thread, NULL) == 0);

  /* Check that `tactor_output` matches expected output. */
  static const float kExpected[8] = {10.0f, 20.0f, 30.0f, 40.0f,
                                     50.0f, 60.0f, 70.0f, 80.0f};
  CHECK(info.num_tactor_output == 8);
  CHECK(!memcmp(info.tactor_output, kExpected, sizeof(kExpected)));

  free(info.tactor_output);
  pthread_mutex_destroy(&info.lock);
  TactilePlayerFree(info.player);
}

int main(int argc, char** argv) {
  TestStreaming();
  TestInterruptedPlayback();
  TestGetRms();
  TestPlaybackThread();

  puts("PASS");
  return EXIT_SUCCESS;
}
