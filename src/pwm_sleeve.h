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
// Arduino-compatible PWM playback library for the sleeve using the HAL layer.
//
// This library could be used for Linear Resonant Actuators (LRA) or voice
// coils. Pwm is passed through a second-order passive low pass filter with
// around 700 Hz cut off.
//
// This filtered signal is passed as single-ended input to MAX98306 audio
// amplifiers. The class-D amplifier output is connected to the tactors.
// Amplifier datasheet here:
// https://www.maximintegrated.com/en/products/analog/audio/MAX98306.html
//
// The sleeve uses 6 audio amplifiers and a total of 12 PWM channels.
// There are 3 PWM modules. Each module has 4 channels.
// The values in each channel can be set independently.
// The PWM uses: 1.5 kB of RAM with 64 PWM values for each channel (3 module * 4
// channels * 64 values * 2 byte each value)
//
// An example snippet for initialization steps are are as following:
// SleeveTactors.Initialize();
// SleeveTactors.SetNumberRepeats(8);
// SleeveTactors.OnSequenceEnd(on_PWM_sequence_end);
// SleeveTactors.StartPlayback();
//
// After initialization, this code  plays individual tactors one after another.
// for(int pwm_module = 0; pwm_module<3; ++pwm_module) {
//   for (int pwm_channel = 0; pwm_channel<4; ++pwm_channel) {
//     SleeveTactors.UpdatePwmModuleChannel(sin_wave_downsample,
//                                          pwm_module,pwm_channel);
//     vTaskDelay(500);  // a delay of 500 ms.
//     SleeveTactors.SilencePwmModuleChannel(pwm_module,pwm_channel);
//   }
// }
//
// The PWM hardware and registers are described here:
// https://infocenter.nordicsemi.com/index.jsp?topic=%2Fcom.nordic.infocenter.nrf52832.ps.v1.1%2Fpwm.html
// HAL is described here:
// https://infocenter.nordicsemi.com/topic/com.nordic.infocenter.sdk5.v12.0.0/group__nrf__pwm__hal.html

#ifndef AUDIO_TO_TACTILE_SRC_PWM_SLEEVE_H_
#define AUDIO_TO_TACTILE_SRC_PWM_SLEEVE_H_

#include <stdint.h>

#include "nrf_pwm.h"  // NOLINT(build/include)

namespace audio_tactile {

// PWM module definitions.
enum {
  kPWMIrqPriority = 7,  // Lowest priority.
  kPwmTopValue = 512,   // The individual PWM values can't be above this number.
  kNumPwmValues = 8,
  kNumChannels = 4,
  kUpsamplingFactor = 8,
  kNumPwmModules = 3,
  kNumTotalPwm = 12
};

// Sleeve PWM definitions.
// The pins on port 1 are always offset by 32. For example pin 7 (P1.07) is 39.
enum {
  kSleevePwmL1Pin = 13,  // On P0.13.
  kSleevePwmR1Pin = 14,  // On P0.14.
  kSleevePwmL2Pin = 17,  // On P0.17.
  kSleevePwmR2Pin = 16,  // On P0.16.
  kSleevePwmL3Pin = 21,  // On P0.21.
  kSleevePwmR3Pin = 32,  // On P0.32.
  kSleevePwmL4Pin = 29,  // On P0.29.
  kSleevePwmR4Pin = 2,   // On P0.02.
  kSleevePwmL5Pin = 31,  // On P0.31.
  kSleevePwmR5Pin = 30,  // On P0.30.
  kSleevePwmL6Pin = 45,  // On P1.13.
  kSleevePwmR6Pin = 42,  // On P1.10.

  kSleeveAmpEnablePin1 = 33,  // On 1.01.
  kSleeveAmpEnablePin2 = 15,  // On 0.15.
  kSleeveAmpEnablePin3 = 8,   // On 0.08.
  kSleeveAmpEnablePin4 = 41,  // On 1.09.
  kSleeveAmpEnablePin5 = 43,  // On 1.11.
  kSleeveAmpEnablePin6 = 19   // On 0.19.
};

class Pwm {
 public:
  Pwm();

  // This function starts the tactors on the sleeve. Also, initializes amplifier
  // pins.
  void Initialize();

  // Upsampling. Each PWM value can be repeated multiple times.
  // 0 means each pwm value is played once, For example, 1,2,3,4
  // 1 means each pwm value is repeated once. For example, 1,1,2,2,3,3,4,4.
  void SetUpsamplingFactor(uint32_t upsampling_factor);

  // Stop the callbacks, disables the PWM.
  void DisablePwm();

  // Start the callbacks, enable the PWM. Both amplifiers and PWM needs to be
  // enabled to produce output to tactors.
  void EnablePwm();

  // Disable all audio amplifiers with a hardware pin. The pwm remains
  // functional.
  void DisableAmplifiers();

  // Enable all audio amplifiers with a hardware pin.
  void EnableAmplifiers();

  // Start the continuous playback.
  // TODO arduino crashes when I call this fxn, but PWM works fine w/o
  // it. Is hal version somehow different or could it be freeRTOS ?
  void StartPlayback();

  // This function is called when the sequence is finished playing.
  void IrqHandler(NRF_PWM_Type* pwm_module, uint8_t which_pwm_module);

  // Set new PWM values in a module. Copies them to the playback buffer.
  // There are 4 channels, written in a continuous block of RAM.
  // The size of the data should be 4 * kNumPwmValues.
  void UpdatePwmModule(uint16_t* new_data, int which_module);

  // Copies new PWM only to a specific channel of a module
  void UpdatePwmModuleChannel(uint16_t* new_data, int which_module,
                              int which_channel);

  // Copies new PWM only to a specific channel of a module, converts from float
  void UpdatePwmModuleChannelFloat(float* new_data, int which_module,
                                   int which_channel);

  // Update all 12 channels.
  // The data array is a byte array. The size is 96 bytes.
  // The samples are in interleaved format: output[c + kNumChannel *
  // n] = nth sample for channels
  // The order is as following. Tactor: (PWM module, PWM channel).
  //  1: (0, 0)    7: (1, 2)
  //  2: (0, 1)    8: (1, 3)
  //  3: (0, 2)    9: (2, 0)
  //  4: (0, 3)    10: (2, 1)
  //  5: (1, 0)    11: (2, 2)
  //  6: (1, 1)    12: (2, 3)
  void UpdatePwmAllChannelsByte(uint8_t* new_data);

  // Sets values of specific channel to zeros, so there is nothing to play.
  void SilencePwmModuleChannel(int which_module, int which_channel);

  // This function is called when sequence is finished.
  void OnSequenceEnd(void (*function)(void));

  // Returns which PWM module triggered the interrupt.
  // 0 - module PWM0.
  // 1 - module PWM1.
  // 2 - module PWM2.
  uint8_t GetEvent();

  // Converts a float to a value pwm can undestand (uint16_t between 0 and
  // kPwmTopValue).
  uint16_t FloatToPwmSample(float sample);

 private:
  // Internal initialization helper.
  void InitializePwmModule(NRF_PWM_Type* p_reg, uint32_t pins[4]);

  // Playback buffer.
  // In "individual" decoder mode, buffer represents 4 channels:
  // <pin 1 PWM 1>, <pin 2 PWM 1>, <pin 3 PWM 1 >, <pin 4 PWM 1>,
  // <pin 1 PWM 2>, <pin 2 PWM 2>, ....
  // Even if we only use two pins, we still need to set values for
  // 4 channels, as easy DMA reads them consecutively.
  // The playback on pin 1 will be <pin 1 PWM 1>, <pin 1 PWM 2>.
  uint16_t pwm_buffer_[kNumPwmModules][kNumPwmValues * kNumChannels];
};

extern Pwm SleeveTactors;

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_PWM_SLEEVE_H_
