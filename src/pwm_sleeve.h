// Copyright 2020-2021 Google LLC
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

// NOLINTBEGIN(build/include)

#include <stdint.h>

#include "board_defs.h"
#include "nrf_pwm.h"
#include "cpp/constants.h"

// NOLINTEND

#if SLEEVE_BOARD || SLIM_BOARD

namespace audio_tactile {

// PWM module definitions.
enum {
  kPWMIrqPriority = 7,  // Lowest priority.
  kPwmTopValue = 512,   // The individual PWM values can't be above this number.
  kUpsamplingFactor = 8,
};

// PWM definitions.
// The pins on port 1 are always offset by 32. For example pin 7 (P1.07) is 39.
#if SLEEVE_BOARD
enum {
  kPwmL1Pin = 13,  // On P0.13.
  kPwmR1Pin = 14,  // On P0.14.
  kPwmL2Pin = 17,  // On P0.17.
  kPwmR2Pin = 16,  // On P0.16.
  kPwmL3Pin = 21,  // On P0.21.
  kPwmR3Pin = 32,  // On P0.32.
  kPwmL4Pin = 29,  // On P0.29.
  kPwmR4Pin = 2,   // On P0.02.
  kPwmL5Pin = 31,  // On P0.31.
  kPwmR5Pin = 30,  // On P0.30.
  kPwmL6Pin = 45,  // On P1.13.
  kPwmR6Pin = 42,  // On P1.10.

  kAmpEnablePin1 = 33,  // On 1.01.
  kAmpEnablePin2 = 15,  // On 0.15.
  kAmpEnablePin3 = 8,   // On 0.08.
  kAmpEnablePin4 = 41,  // On 1.09.
  kAmpEnablePin5 = 43,  // On 1.11.
  kAmpEnablePin6 = 19   // On 0.19.
};
#endif

#if SLIM_BOARD
enum {
  kPwmL1Pin = 13,  // On P0.13.
  kPwmR1Pin = 14,  // On P0.14.
  kPwmL2Pin = 46,  // On P1.14.
  kPwmR2Pin = 12,  // On P0.12.
  kPwmL3Pin = 47,  // On P1.15.
  kPwmR3Pin = 32,  // On P1.00.
  kPwmL4Pin = 40,  // On P1.08.
  kPwmR4Pin = 41,  // On P1.09.
  kPwmL5Pin = 11,  // On P0.11.
  kPwmR5Pin = 16,  // On P0.16.
  kPwmL6Pin = 27,  // On P0.27.
  kPwmR6Pin = 15,  // On P0.15.

  kAmpEnablePin1 = 33,  // On 1.01.
  kAmpEnablePin2 = 44,  // On 1.12.
  kAmpEnablePin3 = 8,   // On 0.08.
  kAmpEnablePin4 = 39,  // On 1.07.
  kAmpEnablePin5 = 43,  // On 1.11.
  kAmpEnablePin6 = 35   // On 1.03.
};
#endif

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
  void StartPlayback();

  // This function is called when the sequence is finished playing.
  void IrqHandler(NRF_PWM_Type* pwm_module, uint8_t which_pwm_module);

  // Set new PWM values in a module. Copies them to the playback buffer.
  // There are 4 channels, written in a continuous block of RAM.
  // The size of the data should be 4 * kNumPwmValues.
  void UpdatePwmModule(const uint16_t* data, int module);

  // In the following, the `channel` arg is a zero-based flat 1D index between 0
  // and 11. At a lower level, channels are driven by PWMs modules, each module
  // driving up to 4 channels. Another variation is that audio channels are
  // conventionally indexed starting from 1. So elsewhere, these channels might
  // be referred to by (module, module-channel) or by 1-based index:
  //
  //   `channel`  (module, module-channel)   1-based index
  //   ---------------------------------------------------
  //          0                     (0, 0)               1
  //          1                     (0, 1)               2
  //          2                     (0, 2)               3
  //          3                     (0, 3)               4
  //          4                     (1, 0)               5
  //          5                     (1, 1)               6
  //          6                     (1, 2)               7
  //          7                     (1, 3)               8
  //          8                     (2, 0)               9
  //          9                     (2, 1)              10
  //         10                     (2, 2)              11
  //         11                     (2, 3)              12

  // Sets values of specific channel to zeros, so there is nothing to play.
  void SilenceChannel(int channel);

  // Copies new PWM only to a specific channel.
  void UpdateChannel(int channel, const uint16_t* data);

  // Same as above, but converts from from float samples in the [-1, 1] range.
  void UpdateChannel(int channel, const float* data);

  // Sets the PWM samples for a specified channel as
  //
  //   ith sample = FloatToPwmSample(gain * data[i * stride])
  //
  // for i = 0, 1, ..., kNumPwmValues - 1. The `stride` is the number of
  // elements between successive reads. No clipping is done.
  void UpdateChannelWithGain(int channel, float gain, const float* data,
                             int stride = 1);

  // Update all 12 channels.
  // The data array is a byte array. The size is 96 bytes.
  // The samples are in interleaved format:
  //   output[c + kNumChannel * n] = nth sample for channels
  // The order is as following. Tactor: (PWM module, PWM channel).
  //  1: (0, 0)    7: (1, 2)
  //  2: (0, 1)    8: (1, 3)
  //  3: (0, 2)    9: (2, 0)
  //  4: (0, 3)    10: (2, 1)
  //  5: (1, 0)    11: (2, 2)
  //  6: (1, 1)    12: (2, 3)
  void UpdatePwmAllChannelsByte(const uint8_t* data);

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
  enum {
    kNumModules = 3,
    kChannelsPerModule = 4,
    kSamplesPerModule = kNumPwmValues * kChannelsPerModule,
  };

  // Internal initialization helper.
  void InitializePwmModule(NRF_PWM_Type* p_reg, uint32_t pins[4]);

  // Gets pointer to the start of `channel` in pwm_buffer_.
  uint16_t* GetChannelPointer(int channel) {
    return pwm_buffer_ +
        kSamplesPerModule * (channel / kChannelsPerModule) +
        (channel % kChannelsPerModule);
  }

  // Playback buffer.
  // In "individual" decoder mode, buffer represents 4 channels:
  // <pin 1 PWM 1>, <pin 2 PWM 1>, <pin 3 PWM 1 >, <pin 4 PWM 1>,
  // <pin 1 PWM 2>, <pin 2 PWM 2>, ....
  // Even if we only use two pins, we still need to set values for
  // 4 channels, as easy DMA reads them consecutively.
  // The playback on pin 1 will be <pin 1 PWM 1>, <pin 1 PWM 2>.
  uint16_t pwm_buffer_[kNumModules * kNumPwmValues * kChannelsPerModule];
};

extern Pwm SleeveTactors;

}  // namespace audio_tactile

#endif  // #if SLEEVE_BOARD || SLIM_BOARD
#endif  // AUDIO_TO_TACTILE_SRC_PWM_SLEEVE_H_
