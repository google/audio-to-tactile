// Copyright 2021 Google LLC
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

#ifndef AUDIO_TO_TACTILE_SRC_CPP_CONSTANTS_H_
#define AUDIO_TO_TACTILE_SRC_CPP_CONSTANTS_H_

namespace audio_tactile {

// Number of ADC samples per buffer.
constexpr int kAdcDataSize = 64;
// Number of PWM samples for each channel per buffer.
constexpr int kNumPwmValues = 8;
// Number of PWM channels.
constexpr int kNumTotalPwm = 12;
// Max length of a TactilePattern pattern string, not including null terminator.
constexpr int kMaxTactilePatternLength = 15;

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_CPP_CONSTANTS_H_
