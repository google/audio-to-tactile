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

#include "pwm_puck.h"  // NOLINT(build/include)

#include <string.h>

#include "look_up.h"     // NOLINT(build/include)
#include "nrf_gpio.h"    // NOLINT(build/include)
#include "nrf_pwm.h"     // NOLINT(build/include)
#include "pwm_common.h"  // NOLINT(build/include)

namespace audio_tactile {

Tactors::Tactors() {}

void Tactors::Initialize() {
  // Configure amplifier shutdown pin.
  nrf_gpio_cfg_output(kPuckAmpEnablePin);
  nrf_gpio_cfg_output(kAnalogSwitch1Pin);
  nrf_gpio_cfg_output(kAnalogSwitch2Pin);

  // Turn on the speaker amplifier and analog switches.
  nrf_gpio_pin_write(kPuckAmpEnablePin, 1);
  nrf_gpio_pin_write(kAnalogSwitch1Pin, 0);
  nrf_gpio_pin_write(kAnalogSwitch2Pin, 0);

  // Enable the PWM.
  nrf_pwm_enable(NRF_PWM0);

  // `kPwmTopValue` is half the number of clock ticks per PWM sample. The PWM
  // sample value should be in [0, kPwmTopValue], and is the clock tick to flip
  // output between high and low (we use NRF_PWM_MODE_UP counter mode).
  // The PWM sample rate is 16 MHz / kPwmTopValue.
  //
  // E.g. with kPwmTopValue = 1024, the sample rate is 15625 Hz.
  nrf_pwm_configure(NRF_PWM0, NRF_PWM_CLK_16MHz, NRF_PWM_MODE_UP_AND_DOWN,
                    kPwmTopValue);

  // Configure the pins.
  uint32_t pins[4] = {kPuckPWMLPin, kPuckPWMRPin, NRF_PWM_PIN_NOT_CONNECTED,
                      NRF_PWM_PIN_NOT_CONNECTED};
  nrf_pwm_pins_set(NRF_PWM0, pins);

  // Enable global interrupts for PWM.
  NVIC_DisableIRQ(PWM0_IRQn);
  NVIC_ClearPendingIRQ(PWM0_IRQn);
  NVIC_SetPriority(PWM0_IRQn, kPWMIrqPriority);
  NVIC_EnableIRQ(PWM0_IRQn);

  // Set the buffer pointers.
  // There are two sequences. We only use sequence 0.
  nrf_pwm_seq_cnt_set(NRF_PWM0, 0, kNumPwmValues * kNumChannels);
  nrf_pwm_seq_ptr_set(NRF_PWM0, 0, pwm_buffer_);

  // Refresh is 1 by default, which means that each PWM pulse is repeated twice.
  // Set it to zero to avoid it. A value of 1 will repeat twice. A 2 will be 3
  // times.
  nrf_pwm_seq_refresh_set(NRF_PWM0, 0, kNumRepeats);

  // Set the decoder. Decoder determines how PWM values are loaded into RAM.
  // We set it to individual, meaning that each value represents a separate pin.
  nrf_pwm_decoder_set(NRF_PWM0, NRF_PWM_LOAD_INDIVIDUAL, NRF_PWM_STEP_AUTO);

  // Enable interrupts.
  nrf_pwm_int_enable(NRF_PWM0, NRF_PWM_INT_SEQSTARTED0_MASK);
  nrf_pwm_int_enable(NRF_PWM0, NRF_PWM_INT_SEQEND0_MASK);
}

void Tactors::StartPlayback() {
  nrf_pwm_task_trigger(NRF_PWM0, NRF_PWM_TASK_SEQSTART0);
}

void Tactors::Disable() {
  nrf_pwm_disable(NRF_PWM0);
  NVIC_DisableIRQ(PWM0_IRQn);
}

void Tactors::Enable() {
  nrf_pwm_enable(NRF_PWM0);
  NVIC_EnableIRQ(PWM0_IRQn);
}

void Tactors::UpdatePwm(uint16_t* data_to_copy, int size) {
  memcpy(pwm_buffer_, data_to_copy,
         kNumPwmValues * kNumChannels * sizeof(int16_t));
}

Tactors PuckTactors;

void Tactors::OnSequenceEnd(void (*function)(void)) {
  on_pwm_sequence_end(function);
}

}  // namespace audio_tactile
