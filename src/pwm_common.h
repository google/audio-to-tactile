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
 *
 *
 * A common pwm driver for both sleeve and the puck. Common driver is used since
 * puck and sleeve share the same pwm modules.
 */

#ifndef AUDIO_TO_TACTILE_SRC_PWM_COMMON_H_
#define AUDIO_TO_TACTILE_SRC_PWM_COMMON_H_

#include "nrf_pwm.h"  // NOLINT(build/include)

#ifdef __cplusplus
extern "C" {
#endif

/*  Common interrupt handler for 3 pwm modules. */
void pwm_irq_handler(NRF_PWM_Type* pwm_module, uint8_t which_pwm_module);

/* Returns which pwm module caused an interrupt.
 * 0 - module PWM0.
 * 1 - module PWM1.
 * 2 - module PWM2.
 */
uint8_t get_pwm_event(void);

/* Callback function for the interrupt handler. Can be assigned a function. */
void on_pwm_sequence_end(void (*function)(void));

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AUDIO_TO_TACTILE_SRC_PWM_COMMON_H_
