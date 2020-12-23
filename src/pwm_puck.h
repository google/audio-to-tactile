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
// PWM playback library for the puck.
//
// It could be used for Linear Resonant Actuators (LRA) or voice coils.
// PWM is passed through a second-order passive low pass filter with around 700
// Hz cut off. This filtered signal is passed as single-ended input to audio
// amplifier (MAX98306). The class-D amplifier output is connected to the
// tactors. The puck has analog switches on the amplifier output; they are used
// for back EMF measurements. The back EMF measurements and analysis will be
// done in a separate library. The switches should remain closed for use
// cases without back EMF.

#ifndef AUDIO_TO_TACTILE_SRC_PWM_PUCK_H_
#define AUDIO_TO_TACTILE_SRC_PWM_PUCK_H_

#include <stdint.h>

namespace audio_tactile {

// PWM module definitions.
enum {
  kPWMIrqPriority = 7,  // lowest priority
  kPwmTopValue = 512,
  kNumPwmValues = 64,
  kNumChannels = 4,
  kNumRepeats = 7
};

// Puck PWM definitions.
// The pins on port 1 are always offset by 32. For example pin 7 (P1.07) is 39.
enum {
  kPuckPWMLPin = 39,       // On P1.07, which maps to 32 + 7 = 39
  kPuckPWMRPin = 38,       // On P1.06, which maps to 32 + 6 = 38
  kPuckAmpEnablePin = 36,  // On 1.04, which maps to 32 + 4 = 36
  kAnalogSwitch1Pin = 14,
  kAnalogSwitch2Pin = 16,
  kPwmChannelsPuck = 2
};

class Tactors {
 public:
  Tactors();

  // This function starts the tactors on the puck.
  void Initialize();

  // Stop the callbacks, disables the PWM.
  void Disable();

  // Start the callbacks, enable the PWM.
  void Enable();

  // Start the continuous playback.
  // TODO: arduino crashes when I call this fxn, but works fine w/o
  // it. Is hal version somehow different or could it be freeRTOS ?
  void StartPlayback();

  // This function is called when the sequence is finished playing.
  void IrqHandler();

  // Set new PWM values. Copies them to the playback buffer.
  void UpdatePwm(uint16_t* new_data, int size);

  // This function is called when sequence is finished.
  void OnSequenceEnd(void (*function)(void));

 private:
  // Playback buffer.
  // In "individual" decoder mode, buffer represents 4 channels:
  // <pin 1 PWM 1>, <pin 2 PWM 1>, <pin 3 PWM 1 >, <pin 4 PWM 1>,
  // <pin 1 PWM 2>, <pin 2 PWM 2>, ....
  // Even if we only use two pins (as here), we still need to set values for
  // 4 channels, as easy DMA reads them consecutively.
  // The playback on pin 1 will be <pin 1 PWM 1>, <pin 1 PWM 2>.
  uint16_t pwm_buffer_[kNumPwmValues * kNumChannels];
};

extern Tactors PuckTactors;

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_PWM_PUCK_H_
