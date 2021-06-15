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
 *
 * Multiple sensors may wish to use the Low Power Comparator for monitoring
 * status, e.g. battery_monitor and temperature_monitor. While only one sensor
 * can be attached at a given time, the lpcomp_common interface allows
 * multiple sensor modules to be written that are compatible with LPCOMP.
 *
 * Each module that wishes to be compatible with LPCOMP should implement
 * the following functions:
 *
 * 1. An initialization function, which enables the comparator, configures
 * the reference threshold, and selects the appropriate pin.
 * (For example, see BatteryMonitor::InitializeLowVoltageInterrupt())
 *
 * 2. A closing function, which disables the comparator and the interrupt.
 * (For example, BatteryMonitor::End())
 *
 * 3. A function which calls on_lpcomp_trigger from lpcomp_common with a
 * callback function that will be run when the interrupt fires.
 * The callback function on the event listener can be defined
 * in the main application to provide feedback to the user, e.g. turning
 * on an LED.
 * (For example, BatteryMonitor::OnLowBatteryEventListener,
 * which is called with LowBatteryWarning() in the .ino )
 *
 * 4. A GetEvent() function that calls get_lpcomp_event from lpcomp_common,
 * which will return the direction of change when the comparator was triggered.
 * (For example, battery_monitor.h GetEvent())
 *
 * LPCOMP_COMP_IRQHandler() will run when the interrupt fires,
 * and will set the event type for get_lpcomp_event as well as
 * run the callback function defined by on_lpcomp_trigger.
 */

#ifndef AUDIO_TO_TACTILE_SRC_LPCOMP_COMMON_H_
#define AUDIO_TO_TACTILE_SRC_LPCOMP_COMMON_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


/* Returns which direction of change caused an interrupt.
0 means lower than reference voltage.
1 means higher than reference voltage. */
uint8_t get_lpcomp_event(void);

/* Callback function for the interrupt handler. Can be assigned a function. */
void on_lpcomp_trigger(void (*function)(void));

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // AUDIO_TO_TACTILE_SRC_LPCOMP_COMMON_H_
