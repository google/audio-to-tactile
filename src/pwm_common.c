/* Copyright 2020 Google LLC
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

#include "pwm_common.h"

static uint8_t pwm_event;

static void (*callback)(void);

void on_pwm_sequence_end(void (*function)(void)) { callback = function; }

uint8_t get_pwm_event() { return pwm_event; }

void PWM0_IRQHandler() { pwm_irq_handler(NRF_PWM0, 0); }
void PWM1_IRQHandler() { pwm_irq_handler(NRF_PWM1, 1); }
void PWM2_IRQHandler() { pwm_irq_handler(NRF_PWM2, 2); }

void pwm_irq_handler(NRF_PWM_Type* pwm_module, uint8_t which_pwm_module) {
  /* Triggered when pwm data is trasfered to RAM with Easy DMA. */
  if (nrf_pwm_event_check(pwm_module, NRF_PWM_EVENT_SEQSTARTED0)) {
    nrf_pwm_event_clear(pwm_module, NRF_PWM_EVENT_SEQSTARTED0);
  }
  /* Triggered after sequence is finished. */
  if (nrf_pwm_event_check(pwm_module, NRF_PWM_EVENT_SEQEND0)) {
    nrf_pwm_event_clear(pwm_module, NRF_PWM_EVENT_SEQEND0);
    pwm_event = which_pwm_module;
    /* Do callback before starting the next sequence, so we can modify the
     * buffer before it gets played.
     */
    callback();
    nrf_pwm_task_trigger(pwm_module, NRF_PWM_TASK_SEQSTART0);
  }
  /* Triggered when playback is stopped. */
  if (nrf_pwm_event_check(pwm_module, NRF_PWM_EVENT_STOPPED)) {
    nrf_pwm_event_clear(pwm_module, NRF_PWM_EVENT_STOPPED);
  }
}
