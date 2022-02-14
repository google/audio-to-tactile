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
 * Benchmarking utilities.
 *
 * This is a set of functions to create a benchmark timer that utilizes one of
 * hardware timers, as described here:
 * https://infocenter.nordicsemi.com/index.jsp?topic=%2Fcom.nordic.infocenter.nrf52832.ps.v1.1%2Ftimer.html
 * The timer is using a HAL provided by Nordic. This timer could be used to
 * measure how long certain functions take to execute BenchmarkTest function
 * provides an example of how the timer could be used.
 *
 * Current time is assigned to instance 1. In some cases the timer instance will
 * need to be assigned to a different timer, as some are used by Bluetooth and
 * other functions. The timer will need to be enabled in the sdk_config file.
 * TIMER_ENABLED 1 and corresponding instance: (e.g., TIMER1_ENABLED 1)
 */

#ifndef AUDIO_TO_TACTILE_EXTRAS_BENCHMARK_BENCHMARKS_H_
#define AUDIO_TO_TACTILE_EXTRAS_BENCHMARK_BENCHMARKS_H_

#include <stdint.h>

/* TODO: Remove dependence on nrf_drv_timer. */
#include "nrf_drv_timer.h"  // NOLINT(build/include)

#ifdef __cplusplus
extern "C" {
#endif

/* Timer instance. If this timer is already used, pick a different one.
 * NRF52840 has 5 timers
 */
static const nrf_drv_timer_t benchmark_timer = NRF_DRV_TIMER_INSTANCE(1);

/* Gets the elapsed time, and clears the timer */
void StartBenchMarkTimer(void);

/* Disables the timer, will need to be reinitialized to restart. */
void StopBenchMarkTimer(void);

/* Gets the elapsed time ticks, and clears the timer */
uint32_t GetBenchMarkTimer(void);

/* Initializes the timer */
void InitiateBenchmarkTimer(void);

/* Conversion to microseconds from ticks. Depends on the timer frequency.
 * I determined this by looking how much time takes for 1 sec delay
 */
int TicksToMicroseconds(uint32_t ticks);

/* Test function for the benchmark timer. It will time a number of iterations of
 * delay function of 1 sec. The function will print results on the console and
 * should be close to 1 sec or 1e6 microseconds. The delay nrf_delay_ms can be
 * replaced with a different function to time it.
 */
void BenchmarkTest(int iterations);


#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* AUDIO_TO_TACTILE_EXTRAS_BENCHMARK_BENCHMARKS_H_ */
