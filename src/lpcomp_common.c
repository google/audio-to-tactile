/* Copyright 2021 Google LLC
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

#include "lpcomp_common.h"
#include "nrf_lpcomp.h"  // NOLINT(build/include)

static uint8_t lpcomp_event;

static void (*callback)(void);

void on_lpcomp_trigger(void (*function)(void)) { callback = function; }

uint8_t get_lpcomp_event(void) { return lpcomp_event; }

void LPCOMP_COMP_IRQHandler(void) {
  // Clear event.
  NRF_LPCOMP->EVENTS_CROSS = 0;

  // Sample the LPCOMP stores its state in the RESULT register.
  NRF_LPCOMP->TASKS_SAMPLE = 1;
  lpcomp_event = NRF_LPCOMP->RESULT;

  callback();
}
