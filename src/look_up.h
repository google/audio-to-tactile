// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
// Sin wave lookup table for testing the tactors.
//
// Values are scaled from 0 to 512. I used this website to generate the wave:
// https://www.daycounter.com/Calculators/Sine-Generator-Calculator.phtml
// Each value is repeated four times, for each pwm channel. This format can be
// loaded directly into easy DMA.

#ifndef AUDIO_TO_TACTILE_SRC_LOOK_UP_H_
#define AUDIO_TO_TACTILE_SRC_LOOK_UP_H_

#include <stdint.h>

// This is the format used by the Easy DMA pwm, where there are 4 channels.
// As each channel is the same, each sin value is repeated 4 times.
// 0 will be high output on the pwm
// I am not declaring it as const, so it can be modified during run-time.
static uint16_t sin_wave[] = {
    256, 256, 256, 256, 281, 281, 281, 281, 306, 306, 306, 306, 330, 330, 330,
    330, 354, 354, 354, 354, 377, 377, 377, 377, 398, 398, 398, 398, 418, 481,
    418, 418, 437, 437, 437, 437, 454, 454, 454, 454, 469, 469, 469, 469, 482,
    482, 482, 482, 493, 493, 493, 493, 501, 501, 501, 501, 507, 507, 507, 507,
    511, 511, 511, 511, 512, 512, 512, 512, 511, 511, 511, 511, 507, 507, 507,
    507, 501, 501, 501, 501, 493, 493, 493, 493, 482, 482, 482, 482, 469, 469,
    469, 469, 454, 454, 454, 454, 437, 437, 437, 437, 418, 418, 418, 418, 398,
    398, 398, 398, 377, 377, 377, 377, 354, 354, 354, 354, 330, 330, 330, 330,
    306, 306, 306, 306, 281, 281, 281, 281, 256, 256, 256, 256, 231, 231, 231,
    231, 206, 206, 206, 206, 182, 182, 182, 182, 158, 158, 158, 158, 135, 135,
    135, 135, 114, 114, 114, 114, 94,  94,  94,  94,  75,  75,  75,  75,  58,
    58,  58,  58,  43,  43,  43,  43,  30,  30,  30,  30,  19,  19,  19,  19,
    11,  11,  11,  11,  5,   5,   5,   5,   1,   1,   1,   1,   0,   0,   0,
    0,   1,   1,   1,   1,   5,   5,   5,   5,   11,  11,  11,  11,  19,  19,
    19,  19,  30,  30,  30,  30,  43,  43,  43,  43,  58,  58,  58,  58,  75,
    75,  75,  75,  94,  94,  94,  94,  114, 114, 114, 114, 135, 135, 135, 135,
    158, 158, 158, 158, 182, 182, 182, 182, 206, 206, 206, 206, 231, 231, 231,
    231};

#endif  // AUDIO_TO_TACTILE_SRC_LOOK_UP_H_
