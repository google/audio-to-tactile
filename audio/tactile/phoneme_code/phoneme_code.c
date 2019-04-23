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

#include "audio/tactile/phoneme_code/phoneme_code.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "audio/tactile/util.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define kNumChannels 24

/* kScale is half the max amplitude, set to match what Purdue did. The max value
 * for legal signals in [-1, 1] is kScale = 0.5f.
 */
static const float kScale = 0.0765f;

/* Evaluate a squared Tukey window, nonzero over 0 < t < `window_duration` and
 * having transitions of length `transition`.
 */
static float Window(float window_duration, float transition, float t) {
  const float result = TukeyWindow(window_duration, transition, t);
  return result * result;
}

/* 60Hz sine wave with a squared Tukey window. */
static float Pulse60Hz(float t, float duration) {
  return sin(2.0 * M_PI * 60.0 * t) * Window(duration, 0.005f, t);
}

/* 300Hz sine wave with a squared Tukey window. */
static float Pulse300Hz(float t, float duration) {
  return sin(2.0 * M_PI * 300.0 * t) * Window(duration, 0.005f, t);
}

/* 300Hz pulse with modulation. Used in B, D, DH, G, I, OO, and others. */
static float ModulatedBuzz(float t, float duration, float modulation_hz) {
  return Pulse300Hz(t, duration) * kScale *
         (1 + 0.5 * sin(2 * M_PI * modulation_hz * t));
}

/* 300Hz pulse with a gentle squared Hann window. Used in OE. */
static float OEPulse(float t) {
  const float kDuration = 0.12f;
  return sin(2.0 * M_PI * 300.0 * t) * Window(kDuration, kDuration / 2, t);
}

/* 60Hz pulse with 4Hz modulation. Pulse used in M, N, and NG. */
static float NasalPhonemeBuzz(float t) {
  return kScale * Pulse60Hz(t, 0.4f) * (1 + sin(2.0 * M_PI * 8 * t));
}

/* Generate AE phoneme code (0 <= t <= 0.4s). "Twinkle" sensation. */
static void PhonemeAE(float t, float* frame) {
  int c;
  for (c = 0; c < 24; ++c) {
    frame[c] = 0.0f;
  }
  const float kDuration = 0.03333f;
  frame[0] = kScale * (Pulse300Hz(t - 0.1332f, kDuration) +
                       Pulse300Hz(t - 0.3332f, kDuration));
  frame[1] = kScale * (Pulse300Hz(t - 0.1f, kDuration) +
                       Pulse300Hz(t - 0.3f, kDuration));
  frame[4] = kScale * (Pulse300Hz(t - 0.1660f, kDuration) +
                       Pulse300Hz(t - 0.3660f, kDuration));
  frame[5] = kScale * (Pulse300Hz(t - 0.0666f, kDuration) +
                       Pulse300Hz(t - 0.2666f, kDuration));
  frame[8] =
      kScale * (Pulse300Hz(t, kDuration) + Pulse300Hz(t - 0.2f, kDuration));
  frame[9] = kScale * (Pulse300Hz(t - 0.0333f, kDuration) +
                       Pulse300Hz(t - 0.2333f, kDuration));
}

/* Generate AH phoneme code (0 <= t <= 0.4s). Wide movement sensation. */
static void PhonemeAH(float t, float* frame) {
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = 0.0f;
  }
  const float kDuration = 0.06667f;
  frame[0] = kScale * Pulse60Hz(t, kDuration);
  frame[1] = frame[0];
  frame[4] = kScale * Pulse60Hz(t - 0.0666f, kDuration);
  frame[5] = frame[4];
  frame[8] = kScale * Pulse60Hz(t - 0.1332f, kDuration);
  frame[9] = frame[8];
  frame[12] = kScale * Pulse60Hz(t - 0.2f, kDuration);
  frame[13] = frame[12];
  frame[16] = kScale * Pulse60Hz(t - 0.2666f, kDuration);
  frame[17] = frame[16];
  frame[20] = kScale * Pulse60Hz(t - 0.3333f, kDuration);
  frame[21] = frame[20];
}

/* Generate AW phoneme code (0 <= t <= 0.4s). "Twinkle" sensation. */
static void PhonemeAW(float t, float* frame) {
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = 0.0f;
  }
  const float kDuration = 0.03333f;
  frame[14] = kScale * (Pulse300Hz(t - 0.1332f, kDuration) +
                        Pulse300Hz(t - 0.3332f, kDuration));
  frame[15] = kScale * (Pulse300Hz(t - 0.1f, kDuration) +
                        Pulse300Hz(t - 0.3f, kDuration));
  frame[18] = kScale * (Pulse300Hz(t - 0.1660f, kDuration) +
                        Pulse300Hz(t - 0.3660f, kDuration));
  frame[19] = kScale * (Pulse300Hz(t - 0.0666f, kDuration) +
                        Pulse300Hz(t - 0.2666f, kDuration));
  frame[22] = kScale * (Pulse300Hz(t, kDuration) +
                        Pulse300Hz(t - 0.2f, kDuration));
  frame[23] = kScale * (Pulse300Hz(t - 0.0333f, kDuration) +
                        Pulse300Hz(t - 0.2333f, kDuration));
}

/* Generate AY phoneme code (0 <= t <= 0.4s). Rumbling sensation. */
static void PhonemeAY(float t, float* frame) {
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = 0.0f;
  }
  const float kDuration = 0.05f;
  const float kModulationHz = 30.0f;
  frame[8] = ModulatedBuzz(t, kDuration, kModulationHz);
  frame[9] = ModulatedBuzz(t - 0.3495f, kDuration, kModulationHz);
  frame[12] = ModulatedBuzz(t - 0.05f, kDuration, kModulationHz);
  frame[13] = ModulatedBuzz(t - 0.3f, kDuration, kModulationHz);
  frame[16] = ModulatedBuzz(t - 0.1f, kDuration, kModulationHz);
  frame[17] = ModulatedBuzz(t - 0.25f, kDuration, kModulationHz);
  frame[20] = ModulatedBuzz(t - 0.15f, kDuration, kModulationHz);
  frame[21] = ModulatedBuzz(t - 0.2f, kDuration, kModulationHz);
}

/* Generate B phoneme code (0 <= t <= 0.14s). */
static void PhonemeB(float t, float* frame) {
  const float buzz = ModulatedBuzz(t, 0.14f, 30.0f);
  int c;
  /* For a loop like this with a static number of iterations, Clang with -O2 or
   * GCC with -O3 will unroll it and evaluate the `c == 16 || c == 17 || ...`
   * conditional at compile time [https://godbolt.org/z/x6wwST].
   */
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 16 || c == 17 || c == 20 || c == 21) ? buzz : 0.0f;
  }
}

/* Generate CH phoneme code (0 <= t <= 0.4s). */
static void PhonemeCH(float t, float* frame) {
  const float buzz =
      kScale * sin(2.0 * M_PI * 300.0 * t) * Window(0.4f, 0.2f, t);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 0 || c == 1 || c == 20 || c == 21) ? buzz : 0.0f;
  }
}

/* Generate D phoneme code (0 <= t <= 0.14s). */
static void PhonemeD(float t, float* frame) {
  const float buzz = ModulatedBuzz(t, 0.14f, 30.0f);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 10 || c == 11 || c == 14 || c == 15) ? buzz : 0.0f;
  }
}

/* Generate DH phoneme code (0 <= t <= 0.4s). */
static void PhonemeDH(float t, float* frame) {
  const float buzz = ModulatedBuzz(t, 0.4f, 8.0f);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 8 || c == 9 || c == 12 || c == 13) ? buzz : 0.0f;
  }
}

/* Generate EE phoneme code (0 <= t <= 0.4s). Smooth movement sensation. */
static void PhonemeEE(float t, float* frame) {
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = 0.0f;
  }
  const float kDuration = 0.09333f;
  frame[0] = kScale * Pulse300Hz(t - 0.3063f, kDuration);
  frame[4] = kScale * Pulse300Hz(t - 0.2448f, kDuration);
  frame[8] = kScale * Pulse300Hz(t - 0.1836f, kDuration);
  frame[12] = kScale * Pulse300Hz(t - 0.1224f, kDuration);
  frame[16] = kScale * Pulse300Hz(t - 0.0612f, kDuration);
  frame[20] = kScale * Pulse300Hz(t, kDuration);
}

/* Generate EH phoneme code (0 <= t <= 0.24s). Grabbing sensation. */
static void PhonemeEH(float t, float* frame) {
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = 0.0f;
  }
  const float kDuration = 0.09603f;
  frame[0] = kScale * Pulse300Hz(t, kDuration);
  frame[1] = frame[0];
  frame[2] = frame[0];
  frame[3] = frame[0];
  frame[12] = kScale * Pulse300Hz(t - 0.14395f, kDuration);
  frame[13] = frame[12];
  frame[14] = frame[12];
  frame[15] = frame[12];
}

/* Generate ER phoneme code (0 <= t <= 0.4s). "Twinkle" sensation. */
static void PhonemeER(float t, float* frame) {
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = 0.0f;
  }
  const float kDuration = 0.03333f;
  frame[2] = kScale * (Pulse300Hz(t - 0.1f, kDuration) +
                       Pulse300Hz(t - 0.3f, kDuration));
  frame[3] = kScale * (Pulse300Hz(t - 0.1332f, kDuration) +
                       Pulse300Hz(t - 0.3332f, kDuration));
  frame[6] = kScale * (Pulse300Hz(t - 0.0666f, kDuration) +
                       Pulse300Hz(t - 0.2666f, kDuration));
  frame[7] = kScale * (Pulse300Hz(t - 0.1660f, kDuration) +
                       Pulse300Hz(t - 0.3660f, kDuration));
  frame[10] = kScale * (Pulse300Hz(t - 0.0333f, kDuration) +
                        Pulse300Hz(t - 0.2333f, kDuration));
  frame[11] = kScale * (Pulse300Hz(t, kDuration) +
                        Pulse300Hz(t - 0.2f, kDuration));
}

/* Generate F phoneme code (0 <= t <= 0.4s). */
static void PhonemeF(float t, float* frame) {
  const float buzz = kScale * Pulse300Hz(t, 0.4f);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c >= 20) ? buzz : 0.0f;
  }
}

/* Generate G phoneme code (0 <= t <= 0.14s). */
static void PhonemeG(float t, float* frame) {
  const float buzz = ModulatedBuzz(t, 0.14f, 30.0f);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 0 || c == 1 || c == 4 || c == 5) ? buzz : 0.0f;
  }
}

/* Generate H phoneme code (0 <= t <= 0.4s). */
static void PhonemeH(float t, float* frame) {
  const float buzz =
      kScale * sin(2.0 * M_PI * 60.0 * t) * Window(0.4f, 0.2f, t);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (12 <= c && c < 20) ? buzz : 0.0f;
  }
}

/* Generate I phoneme code (0 <= t <= 0.4s). Rumbling sensation. */
static void PhonemeI(float t, float* frame) {
  const float kDuration = 0.05f;
  const float kModulationHz = 30.0f;
  frame[0] = ModulatedBuzz(t - 0.3495f, kDuration, kModulationHz);
  frame[1] = frame[0];
  frame[2] = ModulatedBuzz(t, kDuration, kModulationHz);
  frame[3] = frame[2];
  frame[4] = ModulatedBuzz(t - 0.3f, kDuration, kModulationHz);
  frame[5] = frame[4];
  frame[6] = ModulatedBuzz(t - 0.05f, kDuration, kModulationHz);
  frame[7] = frame[6];
  frame[8] = ModulatedBuzz(t - 0.25f, kDuration, kModulationHz);
  frame[9] = frame[8];
  frame[10] = ModulatedBuzz(t - 0.1f, kDuration, kModulationHz);
  frame[11] = frame[10];
  frame[12] = ModulatedBuzz(t - 0.2f, kDuration, kModulationHz);
  frame[13] = frame[12];
  frame[14] = ModulatedBuzz(t - 0.15f, kDuration, kModulationHz);
  frame[15] = frame[14];
  int c;
  for (c = 16; c < kNumChannels; ++c) {
    frame[c] = 0.0f;
  }
}

/* Generate IH phoneme code (0 <= t <= 0.24s). Quick smooth movement. */
static void PhonemeIH(float t, float* frame) {
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    if (c != 0 && c != 4 && c != 8 && c != 12) {
      frame[c] = 0.0f;
    }
  }
  const float kDuration = 0.06f;
  frame[0] = kScale * Pulse300Hz(t, kDuration);
  frame[4] = kScale * Pulse300Hz(t - 0.06f, kDuration);
  frame[8] = kScale * Pulse300Hz(t - 0.12f, kDuration);
  frame[12] = kScale * Pulse300Hz(t - 0.18f, kDuration);
}

/* Generate J phoneme code (0 <= t <= 0.4s). */
static void PhonemeJ(float t, float* frame) {
  const float buzz =
      Pulse300Hz(t, 0.4f) *
      (0.0155 + (0.1375 - 0.0155) * (1 + sin(2 * M_PI * 8.0 * t)) / 2);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 0 || c == 1 || c == 20 || c == 21) ? buzz : 0.0f;
  }
}

/* Generate K phoneme code (0 <= t <= 0.14s). */
static void PhonemeK(float t, float* frame) {
  const float buzz = kScale * Pulse300Hz(t, 0.14f);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 0 || c == 1 || c == 4 || c == 5) ? buzz : 0.0f;
  }
}

/* Generate L phoneme code (0 <= t <= 0.4s). */
static void PhonemeL(float t, float* frame) {
  const float buzz = ModulatedBuzz(t, 0.4f, 30.0f);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 18 || c == 19 || c == 22 || c == 23) ? buzz : 0.0f;
  }
}

/* Generate M phoneme code (0 <= t <= 0.4s). */
static void PhonemeM(float t, float* frame) {
  const float buzz = NasalPhonemeBuzz(t);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 16 || c == 17 || c == 20 || c == 21) ? buzz : 0.0f;
  }
}

/* Generate N phoneme code (0 <= t <= 0.4s). */
static void PhonemeN(float t, float* frame) {
  const float buzz = NasalPhonemeBuzz(t);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 10 || c == 11 || c == 14 || c == 15) ? buzz : 0.0f;
  }
}

/* Generate NG phoneme code (0 <= t <= 0.4s). */
static void PhonemeNG(float t, float* frame) {
  const float buzz = NasalPhonemeBuzz(t);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 0 || c == 1 || c == 4 || c == 5) ? buzz : 0.0f;
  }
}

/* Generate OE phoneme code (0 <= t <= 0.4s). Smooth circular ring. */
static void PhonemeOE(float t, float* frame) {
  int c;
  for (c = 0; c < 8; ++c) {
    frame[c] = 0.0f;
  }
  frame[8] = kScale * (OEPulse(t) + OEPulse(t - 0.28f));
  frame[9] = kScale * OEPulse(t - 0.07f);
  frame[10] = kScale * OEPulse(t - 0.14f);
  frame[11] = kScale * OEPulse(t - 0.21f);
  frame[12] = frame[8];
  frame[13] = frame[9];
  frame[14] = frame[10];
  frame[15] = frame[11];
  for (c = 16; c < kNumChannels; ++c) {
    frame[c] = 0.0f;
  }
}

/* Generate OO phoneme code (0 <= t <= 0.4s). Rumbling sensation. */
static void PhonemeOO(float t, float* frame) {
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = 0.0f;
  }
  const float kDuration = 0.0667f;
  const float kModulationHz = 30.0f;
  frame[2] = ModulatedBuzz(t - 0.3333f, kDuration, kModulationHz);
  frame[3] = frame[2];
  frame[6] = ModulatedBuzz(t - 0.2666f, kDuration, kModulationHz);
  frame[7] = frame[6];
  frame[10] = ModulatedBuzz(t - 0.2f, kDuration, kModulationHz);
  frame[11] = frame[10];
  frame[14] = ModulatedBuzz(t - 0.1333f, kDuration, kModulationHz);
  frame[15] = frame[14];
  frame[18] = ModulatedBuzz(t - 0.0666f, kDuration, kModulationHz);
  frame[19] = frame[18];
  frame[22] = ModulatedBuzz(t, kDuration, kModulationHz);
  frame[23] = frame[22];
}

/* Generate OW phoneme code (0 <= t <= 0.4s). Tapping sensation. */
static void PhonemeOW(float t, float* frame) {
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = 0.0f;
  }
  const float kDuration = 0.024f;
  frame[4] =
      kScale *
      (Pulse300Hz(t - 0.2824f, kDuration) + Pulse300Hz(t - 0.3295f, kDuration) +
       Pulse300Hz(t - 0.3767f, kDuration)) *
      Window(0.4f, 0.005f, t);
  frame[12] = kScale * (Pulse300Hz(t - 0.1412f, kDuration) +
                        Pulse300Hz(t - 0.1888f, kDuration) +
                        Pulse300Hz(t - 0.2360f, kDuration));
  frame[20] =
      kScale * (Pulse300Hz(t, kDuration) + Pulse300Hz(t - 0.0472f, kDuration) +
                Pulse300Hz(t - 0.0939f, kDuration));
}

/* Generate OY phoneme code (0 <= t <= 0.4s). Tapping sensation. */
static void PhonemeOY(float t, float* frame) {
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = 0.0f;
  }
  const float kDuration = 0.024f;
  frame[6] =
      kScale * (Pulse300Hz(t, kDuration) + Pulse300Hz(t - 0.0472f, kDuration) +
                Pulse300Hz(t - 0.0939f, kDuration));
  frame[14] = kScale * (Pulse300Hz(t - 0.1412f, kDuration) +
                        Pulse300Hz(t - 0.1888f, kDuration) +
                        Pulse300Hz(t - 0.2360f, kDuration));
  frame[22] =
      kScale *
      (Pulse300Hz(t - 0.2824f, kDuration) + Pulse300Hz(t - 0.3295f, kDuration) +
       Pulse300Hz(t - 0.3767f, kDuration)) *
      Window(0.4f, 0.005f, t);
}

/* Generate P phoneme code (0 <= t <= 0.14s). */
static void PhonemeP(float t, float* frame) {
  const float buzz = kScale * Pulse300Hz(t, 0.14f);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 16 || c == 17 || c == 20 || c == 21) ? buzz : 0.0f;
  }
}

/* Generate R phoneme code (0 <= t <= 0.4s). */
static void PhonemeR(float t, float* frame) {
  const float buzz = ModulatedBuzz(t, 0.4f, 30.0f);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 2 || c == 3 || c == 6 || c == 7) ? buzz : 0.0f;
  }
}

/* Generate S phoneme code (0 <= t <= 0.4s). */
static void PhonemeS(float t, float* frame) {
  const float buzz = kScale * Pulse300Hz(t, 0.4f);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c < 4) ? buzz : 0.0f;
  }
}

/* Generate SH phoneme code (0 <= t <= 0.4s). */
static void PhonemeSH(float t, float* frame) {
  const float buzz = kScale * Pulse300Hz(t, 0.4f);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 18 || c == 19 || c == 22 || c == 23) ? buzz : 0.0f;
  }
}

/* Generate T phoneme code (0 <= t <= 0.14s). */
static void PhonemeT(float t, float* frame) {
  const float buzz = kScale * Pulse300Hz(t, 0.14f);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 10 || c == 11 || c == 14 || c == 15) ? buzz : 0.0f;
  }
}

/* Generate TH phoneme code (0 <= t <= 0.4s). */
static void PhonemeTH(float t, float* frame) {
  const float buzz = kScale * Pulse300Hz(t, 0.4f);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 8 || c == 9 || c == 12 || c == 13) ? buzz : 0.0f;
  }
}

/* Generate UH phoneme code (0 <= t <= 0.24s). Grabbing sensation. */
static void PhonemeUH(float t, float* frame) {
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = 0.0f;
  }
  const float kDuration = 0.09604f;
  frame[8] = kScale * Pulse300Hz(t - 0.1439f, kDuration);
  frame[9] = frame[8];
  frame[10] = frame[8];
  frame[11] = frame[8];
  frame[20] = kScale * Pulse300Hz(t, kDuration);
  frame[21] = frame[20];
  frame[22] = frame[20];
  frame[23] = frame[20];
}

/* Generate UU phoneme code (0 <= t <= 0.24s). Rumbling sensation. */
static void PhonemeUU(float t, float* frame) {
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = 0.0f;
  }
  const float kDuration = 0.06f;
  const float kModulationHz = 30.0f;
  frame[2] = ModulatedBuzz(t, kDuration, kModulationHz);
  frame[3] = frame[2];
  frame[6] = ModulatedBuzz(t - 0.06f, kDuration, kModulationHz);
  frame[7] = frame[6];
  frame[10] = ModulatedBuzz(t - 0.12f, kDuration, kModulationHz);
  frame[11] = frame[10];
  frame[14] = ModulatedBuzz(t - 0.18f, kDuration, kModulationHz);
  frame[15] = frame[14];
}

/* Generate V phoneme code (0 <= t <= 0.4s). */
static void PhonemeV(float t, float* frame) {
  const float buzz = ModulatedBuzz(t, 0.4f, 8.0f);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c >= 20) ? buzz : 0.0f;
  }
}

/* Generate W phoneme code (0 <= t <= 0.4s). */
static void PhonemeW(float t, float* frame) {
  const float buzz = NasalPhonemeBuzz(t);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c % 2 == 0 && c >= 8) ? buzz : 0.0f;
  }
}

/* Generate Y phoneme code (0 <= t <= 0.4s). */
static void PhonemeY(float t, float* frame) {
  const float buzz = kScale * Pulse60Hz(t, 0.4f);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c % 2 == 0 && c >= 8) ? buzz : 0.0f;
  }
}

/* Generate Z phoneme code (0 <= t <= 0.4s). */
static void PhonemeZ(float t, float* frame) {
  const float buzz = ModulatedBuzz(t, 0.4f, 8.0f);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c < 4) ? buzz : 0.0f;
  }
}

/* Generate ZH phoneme code (0 <= t <= 0.4s). */
static void PhonemeZH(float t, float* frame) {
  const float buzz = ModulatedBuzz(t, 0.4f, 8.0f);
  int c;
  for (c = 0; c < kNumChannels; ++c) {
    frame[c] = (c == 2 || c == 3 || c == 6 || c == 7) ? buzz : 0.0f;
  }
}

const PhonemeCode kPhonemeCodebook[] = {
    /* Fields: phoneme, fun, duration. */
    {"AE", PhonemeAE, 0.4f},  {"AH", PhonemeAH, 0.4f}, {"AW", PhonemeAW, 0.4f},
    {"AY", PhonemeAY, 0.4f},  {"B", PhonemeB, 0.14f},  {"CH", PhonemeCH, 0.4f},
    {"D", PhonemeD, 0.14f},   {"DH", PhonemeDH, 0.4f}, {"EE", PhonemeEE, 0.4f},
    {"EH", PhonemeEH, 0.24f}, {"ER", PhonemeER, 0.4f}, {"F", PhonemeF, 0.4f},
    {"G", PhonemeG, 0.14f},   {"H", PhonemeH, 0.4f},   {"I", PhonemeI, 0.4f},
    {"IH", PhonemeIH, 0.24f}, {"J", PhonemeJ, 0.4f},   {"K", PhonemeK, 0.14f},
    {"L", PhonemeL, 0.4f},    {"M", PhonemeM, 0.4f},   {"N", PhonemeN, 0.4f},
    {"NG", PhonemeNG, 0.4f},  {"OE", PhonemeOE, 0.4f}, {"OO", PhonemeOO, 0.4f},
    {"OW", PhonemeOW, 0.4f},  {"OY", PhonemeOY, 0.4f}, {"P", PhonemeP, 0.14f},
    {"R", PhonemeR, 0.4f},    {"S", PhonemeS, 0.4f},   {"SH", PhonemeSH, 0.4f},
    {"T", PhonemeT, 0.14f},   {"TH", PhonemeTH, 0.4f}, {"UH", PhonemeUH, 0.24f},
    {"UU", PhonemeUU, 0.24f}, {"V", PhonemeV, 0.4f},   {"W", PhonemeW, 0.4f},
    {"Y", PhonemeY, 0.4f},    {"Z", PhonemeZ, 0.4f},   {"ZH", PhonemeZH, 0.4f},
};
const int kPhonemeCodebookSize =
    sizeof(kPhonemeCodebook) / sizeof(*kPhonemeCodebook);

const PhonemeCode* PhonemeCodeByName(const char* name) {
  /* Convert `name` to uppercase, stopping at the first non-alphanumeric char.
   * Since all phoneme names are at most 2 chars, we stop and return NULL if
   * the result would be longer than that.
   */
  char phoneme[3];
  int i;
  for (i = 0; i < 3; ++i) {
    if (!isalnum(name[i])) {
      phoneme[i] = '\0';
      break;
    } else if (i < 2) {
      phoneme[i] = toupper(name[i]);
    } else {
      return NULL; /* Name longer than 2 chars is invalid. */
    }
  }

  /* Find and return codebook entry with matching phoneme name. */
  for (i = 0; i < kPhonemeCodebookSize; ++i) {
    if (!strcmp(phoneme, kPhonemeCodebook[i].phoneme)) {
      return &kPhonemeCodebook[i];
    }
  }
  return NULL; /* Not found. */
}

/* Finds start of next phoneme or NULL in a comma-delimited phonemes string. */
static const char* NextPhoneme(const char* phonemes) {
  phonemes = strchr(phonemes, ',');
  if (phonemes) {
    ++phonemes;
  } /* Increment past the comma. */
  return phonemes;
}

/* Computes the length in seconds for a phonemes string. */
static float PhonemeSignalLength(const char* phonemes, float spacing) {
  float t = 0.0f;
  float length = 0.0f;
  const char* p;
  int num_phonemes = 0;

  for (p = phonemes; p; p = NextPhoneme(p)) {
    const PhonemeCode* signal = PhonemeCodeByName(p);
    if (signal == NULL) {
      return -1.0f;
    }
    if (num_phonemes > 0) {
      t += spacing;
    }
    /* Force nonnegative t, in case `spacing` is negative. */
    if (t < 0.0f) {
      t = 0.0f;
    }
    t += signal->duration;
    if (t > length) {
      length = t; /* Compute length as the maximum value of t. */
    }
    ++num_phonemes;
  }

  return length;
}

float* GeneratePhonemeSignal(const char* phonemes, float spacing,
                             const char* emphasized_phoneme,
                             float emphasis_gain, float sample_rate_hz,
                             int* num_frames) {
  const float length = PhonemeSignalLength(phonemes, spacing);
  if (length < 0.0f) {
    return NULL;
  }

  *num_frames = (int)(sample_rate_hz * length + 0.5f);
  /* Allocate output samples. */
  float* samples = (float*)malloc(kNumChannels * *num_frames * sizeof(float));
  if (samples == NULL) {
    return NULL;
  }

  int i;
  for (i = 0; i < kNumChannels * *num_frames; ++i) {
    samples[i] = 0.0f;
  }

  const int spacing_num_frames = (int)(sample_rate_hz * spacing);
  int write_frame = 0;
  float frame[kNumChannels];

  for (; phonemes; phonemes = NextPhoneme(phonemes)) {
    const PhonemeCode* signal = PhonemeCodeByName(phonemes);

    float gain = 1.0f;
    if (emphasized_phoneme && !strcmp(signal->phoneme, emphasized_phoneme)) {
      gain = emphasis_gain;
    }

    int phoneme_num_frames = (int)(sample_rate_hz * signal->duration + 0.5f);
    /* Make sure that phoneme signal stays within the allocated array. */
    if (phoneme_num_frames > *num_frames - write_frame) {
      phoneme_num_frames = *num_frames - write_frame;
    }

    /* Output the signal for one phoneme. */
    float* dest = samples + kNumChannels * write_frame;
    for (i = 0; i < phoneme_num_frames; ++i, dest += kNumChannels) {
      const float t = i / sample_rate_hz;
      signal->fun(t, frame);
      int c;
      for (c = 0; c < kNumChannels; ++c) {
        dest[c] += gain * frame[c];
      }
    }

    write_frame += phoneme_num_frames + spacing_num_frames;
    /* Force nonnegative write_frame, in case `spacing` is negative. */
    if (write_frame < 0) {
      write_frame = 0;
    }
  }

  return samples;
}

int PhonemeStringIsValid(const char* phonemes) {
  for (; phonemes; phonemes = NextPhoneme(phonemes)) {
    if (PhonemeCodeByName(phonemes) == NULL) {
      return 0;
    }
  }
  return 1;
}
