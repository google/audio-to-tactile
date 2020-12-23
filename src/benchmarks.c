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

#include "benchmarks.h"
#include "nrf_drv_timer.h"
#include "nrf_delay.h"

/* Empty timer handler, required by the timer initialization function */
static void BenchmarkTimerEventHandler(nrf_timer_event_t event_type,
                                       void* p_context)
{
}

void StartBenchMarkTimer() {
  nrf_drv_timer_enable(&benchmark_timer);
}

void StopBenchMarkTimer() {
  nrf_drv_timer_disable(&benchmark_timer);
}

uint32_t GetBenchMarkTimer() {
  uint32_t elapsed_time = nrf_drv_timer_capture(&benchmark_timer,
                                                NRF_TIMER_CC_CHANNEL1);
  nrf_drv_timer_clear(&benchmark_timer);
  return elapsed_time;
}

void InitiateBenchmarkTimer() {
  uint32_t err_code;
  nrf_drv_timer_config_t timer_cfg = NRF_DRV_TIMER_DEFAULT_CONFIG;
  timer_cfg.frequency = NRF_TIMER_FREQ_16MHz;
  timer_cfg.bit_width = NRF_TIMER_BIT_WIDTH_32;
  timer_cfg.mode = NRF_TIMER_MODE_TIMER;
  err_code = nrf_drv_timer_init(&benchmark_timer, &timer_cfg,
                                BenchmarkTimerEventHandler);
  APP_ERROR_CHECK(err_code);
}

int TicksToMicroseconds(uint32_t ticks){
  // 16014798 ticks per 1 seconds or 1e6 microseconds (at 16 Mhz), determined
  // by experiment
  double tick_to_us = (double)(ticks*(1000000.0f)) / 16014798.0f;
  return tick_to_us;
}

void BenchmarkTest(int iterations) {
  InitiateBenchmarkTimer();
  StartBenchMarkTimer();

  uint32_t ticks_array[iterations];

  for (;;) {
    for (int i = 0; i < iterations; ++i) {
      //test using delay function, or replace with your own function
      nrf_delay_ms(1000);
      ticks_array[i] = GetBenchMarkTimer();
    }

    for (int g = 0; g < iterations; ++g) {
      printf("[%d], %d ticks, %d us\n\r", g, ticks_array[g],
             TicksToMicroseconds(ticks_array[g]));
    }
  }
}
