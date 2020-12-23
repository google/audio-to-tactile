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
#include <stdlib.h>
#include <string.h>

struct TactilePlayer {
  float sample_rate_hz;
  int num_channels;
  int fade_out_frames;

  pthread_mutex_t lock;
  float* samples;  /* Guarded by `lock`. */
  int num_frames;  /* Guarded by `lock`. */
  int read_frame;  /* Guarded by `lock`. */
};

TactilePlayer* TactilePlayerMake(int num_channels, float sample_rate_hz) {
  TactilePlayer* player = (TactilePlayer*)malloc(sizeof(TactilePlayer));
  if (player == NULL || pthread_mutex_init(&player->lock, NULL) != 0) {
    free(player);
    return NULL;
  }

  player->sample_rate_hz = sample_rate_hz;
  player->num_channels = num_channels;
  player->fade_out_frames = (int)(0.005f * sample_rate_hz + 0.5f);

  player->samples = NULL;
  player->num_frames = 0;
  player->read_frame = 0;
  return player;
}

void TactilePlayerFree(TactilePlayer* player) {
  if (player != NULL) {
    free(player->samples);
    pthread_mutex_destroy(&player->lock);
  }
  free(player);
}

/* Multiplies `src` with linear fade out ramp of length `num_frames` and adds to
 * `dest`. Assumes `src` and `dest` both have at least `num_frames` frames.
 */
static void AddLinearFadeOut(const float* src, int num_frames, int num_channels,
                             float* dest) {
  if (num_frames <= 0) {
    return;
  }
  const float weight_step = 1.0f / num_frames;
  const float offset = weight_step * (num_frames - 0.5f);
  int i;
  for (i = 0; i < num_frames; ++i) {
    const float weight = offset - i * weight_step;
    int c;
    for (c = 0; c < num_channels; ++c) {
      dest[c] += weight * src[c];
    }

    dest += num_channels;
    src += num_channels;
  }
}

void TactilePlayerPlay(TactilePlayer* player, float* samples, int num_frames) {
  pthread_mutex_lock(&player->lock);
  float* prev_samples = player->samples;

  if (player->read_frame < player->num_frames) {
    /* We are interrupting playback of the previous signal. To avoid clicks,
     * fade out the previous signal and add it to the new signal.
     */
    int count = player->num_frames - player->read_frame;
    if (count > num_frames) {
      count = num_frames;
    }
    if (count > player->fade_out_frames) {
      count = player->fade_out_frames;
    }

    const float* src = prev_samples + player->num_channels * player->read_frame;
    AddLinearFadeOut(src, count, player->num_channels, samples);
  }

  player->samples = samples;
  player->num_frames = num_frames;
  player->read_frame = 0;
  pthread_mutex_unlock(&player->lock);

  free(prev_samples);
}

int TactilePlayerIsActive(TactilePlayer* player) {
  pthread_mutex_lock(&player->lock);
  int is_active = player->read_frame < player->num_frames;
  pthread_mutex_unlock(&player->lock);
  return is_active;
}

int TactilePlayerFillBuffer(TactilePlayer* player, int num_frames,
                            float* output) {
  const int num_channels = player->num_channels;

  pthread_mutex_lock(&player->lock);
  int count = player->num_frames - player->read_frame;
  if (num_frames < count) {
    count = num_frames;
  }

  memcpy(output, player->samples + num_channels * player->read_frame,
         num_channels * count * sizeof(float));

  player->read_frame += count;
  pthread_mutex_unlock(&player->lock);

  output += num_channels * count;
  int remaining_samples = num_channels * (num_frames - count);
  int i;
  for (i = 0; i < remaining_samples; ++i) {
    output[i] = 0.0f;
  }

  return count;
}

void TactilePlayerGetRms(TactilePlayer* player, float window_duration_s,
                         float* rms) {
  const int num_channels = player->num_channels;
  int count = 0;
  int c;
  for (c = 0; c < num_channels; ++c) {
    rms[c] = 0.0f;
  }

  pthread_mutex_lock(&player->lock);

  const int read_frame = player->read_frame;
  if (read_frame < player->num_frames) { /* Playback is active. */
    /* Make a window centered around the current read frame. */
    const int radius = (int)(window_duration_s * player->sample_rate_hz + 0.5f);
    int i = read_frame - radius;
    int i_end = read_frame + radius + 1;

    /* Clamp the window limits to the sample range. */
    if (i < 0) {
      i = 0;
    }
    if (i_end > player->num_frames) {
      i_end = player->num_frames;
    }

    count = i_end - i;
    const float* src = player->samples + num_channels * i;
    for (; i < i_end; ++i, src += num_channels) {
      for (c = 0; c < num_channels; ++c) {
        const float value = src[c];
        rms[c] += value * value; /* Aggregate sum of squares. */
      }
    }
  }

  pthread_mutex_unlock(&player->lock);

  if (count > 0) {
    for (c = 0; c < num_channels; ++c) {
      rms[c] = sqrt(rms[c] / count); /* Convert sum of squares to RMS. */
    }
  }
}
