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

#include "pwm_sleeve.h"  // NOLINT(build/include)

#include <string.h>

#include "board_defs.h"  // NOLINT(build/include)
#include "look_up.h"     // NOLINT(build/include)
#include "nrf_gpio.h"    // NOLINT(build/include)
#include "nrf_pwm.h"     // NOLINT(build/include)
#include "pwm_common.h"  // NOLINT(build/include)

namespace audio_tactile {

Pwm::Pwm() {}

void Pwm::Initialize() {
  // Configure amplifiers shutdowns pin.
  nrf_gpio_cfg_output(kSleeveAmpEnablePin1);
  nrf_gpio_cfg_output(kSleeveAmpEnablePin2);
  nrf_gpio_cfg_output(kSleeveAmpEnablePin3);
  nrf_gpio_cfg_output(kSleeveAmpEnablePin4);
  nrf_gpio_cfg_output(kSleeveAmpEnablePin5);
  nrf_gpio_cfg_output(kSleeveAmpEnablePin6);

  // Turn on the speaker amplifiers.
  EnableAmplifiers();

  // Configure the pins.
  uint32_t pins_pwm0[4] = {kSleevePwmL1Pin, kSleevePwmR1Pin, kSleevePwmL2Pin,
                           kSleevePwmR2Pin};
  uint32_t pins_pwm1[4] = {kSleevePwmL3Pin, kSleevePwmR3Pin, kSleevePwmL4Pin,
                           kSleevePwmR4Pin};
  uint32_t pins_pwm2[4] = {kSleevePwmL5Pin, kSleevePwmR5Pin, kSleevePwmL6Pin,
                           kSleevePwmR6Pin};

  InitializePwmModule(NRF_PWM0, pins_pwm0);
  InitializePwmModule(NRF_PWM1, pins_pwm1);
  InitializePwmModule(NRF_PWM2, pins_pwm2);

  // Set the buffer pointers. Need to set it before running PWM.
  // Tricky part here is that the buffer is always represents 4 channels:
  // <pin 1 PWM> <pin 2 PWM> <pin 3 PWM> <pin 4 PWM> ... <pin 1 PWM>
  // Even if we only use two pins (as here), we still need to set values for
  // 4 channels, as easy DMA reads them consecutively.
  nrf_pwm_seq_cnt_set(NRF_PWM0, 0, kNumPwmValues * kNumChannels);
  nrf_pwm_seq_ptr_set(NRF_PWM0, 0, pwm_buffer_[0]);
  nrf_pwm_seq_cnt_set(NRF_PWM1, 0, kNumPwmValues * kNumChannels);
  nrf_pwm_seq_ptr_set(NRF_PWM1, 0, pwm_buffer_[1]);
  nrf_pwm_seq_cnt_set(NRF_PWM2, 0, kNumPwmValues * kNumChannels);
  nrf_pwm_seq_ptr_set(NRF_PWM2, 0, pwm_buffer_[2]);

  // Enable global interrupts for PWM.
  NVIC_SetPriority(PWM0_IRQn, kPWMIrqPriority);
  NVIC_EnableIRQ(PWM0_IRQn);
  NVIC_SetPriority(PWM1_IRQn, kPWMIrqPriority);
  NVIC_EnableIRQ(PWM1_IRQn);
  NVIC_SetPriority(PWM2_IRQn, kPWMIrqPriority);
  NVIC_EnableIRQ(PWM2_IRQn);
}

void Pwm::InitializePwmModule(NRF_PWM_Type* pwm_module, uint32_t pins[4]) {
  // Enable the PWM.
  nrf_pwm_enable(pwm_module);

  // Configure the pins.
  nrf_pwm_pins_set(pwm_module, pins);

  // `kPwmTopValue` is half the number of clock ticks per PWM sample. The PWM
  // sample value should be in [0, kPwmTopValue], and is the clock tick to flip
  // output between high and low (we use NRF_PWM_MODE_UP counter mode).
  // The PWM sample rate is 16 MHz / kPwmTopValue.
  //
  // E.g. with kPwmTopValue = 1024, the sample rate is 15625 Hz.
  nrf_pwm_configure(pwm_module, NRF_PWM_CLK_8MHz, NRF_PWM_MODE_UP,
                    kPwmTopValue);

  // Refresh is 1 by default, which means that each PWM pulse is repeated twice.
  // Set it to zero to avoid repeats. Also can be set whatever with kNumRepeats.
  nrf_pwm_seq_refresh_set(pwm_module, 0, kUpsamplingFactor);

  // Set the decoder. Decoder determines how PWM values are loaded into RAM.
  // We set it to individual, meaning that each value represents a separate pin.
  nrf_pwm_decoder_set(pwm_module, NRF_PWM_LOAD_INDIVIDUAL, NRF_PWM_STEP_AUTO);

  // Enable interrupts.
  nrf_pwm_int_enable(pwm_module, NRF_PWM_INT_SEQSTARTED0_MASK);
  nrf_pwm_int_enable(pwm_module, NRF_PWM_INT_SEQEND0_MASK);
}

void Pwm::StartPlayback() {
  nrf_pwm_task_trigger(NRF_PWM0, NRF_PWM_TASK_SEQSTART0);
  nrf_pwm_task_trigger(NRF_PWM1, NRF_PWM_TASK_SEQSTART0);
  nrf_pwm_task_trigger(NRF_PWM2, NRF_PWM_TASK_SEQSTART0);
}

void Pwm::SetUpsamplingFactor(uint32_t upsampling_factor) {
  // Subtract 1, since when refresh is at 0, 1 cycle is repeated.
  // Refresh of 1, actually means 2 cycles.
  nrf_pwm_seq_refresh_set(NRF_PWM0, 0, upsampling_factor - 1);
  nrf_pwm_seq_refresh_set(NRF_PWM1, 0, upsampling_factor - 1);
  nrf_pwm_seq_refresh_set(NRF_PWM2, 0, upsampling_factor - 1);
}

void Pwm::DisablePwm() {
  nrf_pwm_disable(NRF_PWM0);
  nrf_pwm_disable(NRF_PWM1);
  nrf_pwm_disable(NRF_PWM2);
  NVIC_DisableIRQ(PWM0_IRQn);
  NVIC_DisableIRQ(PWM1_IRQn);
  NVIC_DisableIRQ(PWM2_IRQn);
}

void Pwm::EnablePwm() {
  nrf_pwm_enable(NRF_PWM0);
  nrf_pwm_enable(NRF_PWM1);
  nrf_pwm_enable(NRF_PWM2);
  NVIC_EnableIRQ(PWM0_IRQn);
  NVIC_EnableIRQ(PWM1_IRQn);
  NVIC_EnableIRQ(PWM2_IRQn);
}

void Pwm::DisableAmplifiers() {
  nrf_gpio_pin_write(kSleeveAmpEnablePin1, 0);
  nrf_gpio_pin_write(kSleeveAmpEnablePin2, 0);
  nrf_gpio_pin_write(kSleeveAmpEnablePin3, 0);
  nrf_gpio_pin_write(kSleeveAmpEnablePin4, 0);
  nrf_gpio_pin_write(kSleeveAmpEnablePin5, 0);
  nrf_gpio_pin_write(kSleeveAmpEnablePin6, 0);
}

void Pwm::EnableAmplifiers() {
  nrf_gpio_pin_write(kSleeveAmpEnablePin1, 1);
  nrf_gpio_pin_write(kSleeveAmpEnablePin2, 1);
  nrf_gpio_pin_write(kSleeveAmpEnablePin3, 1);
  nrf_gpio_pin_write(kSleeveAmpEnablePin4, 1);
  nrf_gpio_pin_write(kSleeveAmpEnablePin5, 1);
  nrf_gpio_pin_write(kSleeveAmpEnablePin6, 1);
}

void Pwm::UpdatePwmModule(uint16_t* data_to_copy, int which_module) {
  memcpy(pwm_buffer_[which_module], data_to_copy,
         kNumPwmValues * kNumChannels * sizeof(int16_t));
}

void Pwm::UpdatePwmModuleChannel(uint16_t* new_data, int which_module,
                                 int which_channel) {
  for (int i = 0; i < kNumPwmValues; ++i) {
    pwm_buffer_[which_module][i * 4 + which_channel] = new_data[i];
  }
}

void Pwm::SilencePwmModuleChannel(int which_module, int which_channel) {
  for (int i = 0; i < kNumPwmValues; ++i) {
    pwm_buffer_[which_module][i * 4 + which_channel] = 0;
  }
}

void Pwm::UpdatePwmModuleChannelFloat(float* new_data, int which_module,
                                      int which_channel) {
  for (int i = 0; i < kNumPwmValues; ++i) {
    uint16_t new_data_int = FloatToPwmSample(new_data[i]);
    pwm_buffer_[which_module][i * 4 + which_channel] = new_data_int;
  }
}

void Pwm::UpdatePwmAllChannelsByte(uint8_t* new_data) {
  uint16_t pwm_channel[kNumPwmValues];
  int array_offset_counter = 0;

  // The values have to be scaled up from a byte to 9-bits (2^9 = 512)
  constexpr int kByteValues = 256;
  constexpr float kScaleFactor = kPwmTopValue / 256;

  for (int module = 0; module < kNumPwmModules; ++module) {
    for (int channel = 0; channel < kNumChannels; ++channel) {
      for (int i = 0; i < kNumPwmValues; ++i) {
        pwm_channel[i] =
            kScaleFactor * new_data[(i * kNumTotalPwm) + array_offset_counter];
      }
      UpdatePwmModuleChannel(pwm_channel, module, channel);
      array_offset_counter = array_offset_counter + 1;
    }
  }
}

uint16_t Pwm::FloatToPwmSample(float sample) {
  constexpr float kScale = 0.5f * kPwmTopValue;
  constexpr float kOffset = kScale + 0.5f;
  return static_cast<uint16_t>(kScale * sample + kOffset);
}

Pwm SleeveTactors;

void Pwm::OnSequenceEnd(void (*function)(void)) {
  on_pwm_sequence_end(function);
}

uint8_t Pwm::GetEvent() { return get_pwm_event(); }

}  // namespace audio_tactile
