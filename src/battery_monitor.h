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
// Battery monitor: callback if battery voltage is low.
//
// This driver provides a callback if the battery voltage is too low, and it can
// measure the battery voltage. The battery voltage is measured using a voltage
// divider. A low power comparator (LPCOMP) hardware module is used to check
// when the battery voltage is low and trigger an interrupt.
// https://infocenter.nordicsemi.com/index.jsp?topic=%2Fcom.nordic.infocenter.nrf52832.ps.v1.1%2Flpcomp.html
// To measure the analog battery voltage, SAADC was used.
// Note: the low power comparator can run only for one of the battery monitor or
// temperature monitor at a time

#ifndef AUDIO_TO_TACTILE_SRC_BATTERY_MONITOR_H_
#define AUDIO_TO_TACTILE_SRC_BATTERY_MONITOR_H_

#include <cstdint>

#include "board_defs.h"     // NOLINT(build/include)
#include "lpcomp_common.h"  // NOLINT(build/include)
#include "nrf_gpio.h"       // NOLINT(build/include)

namespace audio_tactile {

class BatteryMonitor {
 public:
  // Configure the battery pin and initialize interrupts.
  // This function starts the listener (interrupt handler) as well.
  void InitializeLowVoltageInterrupt();

  // Stop the comparator for the battery monitor.
  void End();

  // Allows the user to add low battery warning in other parts of firmware.
  void OnLowBatteryEventListener(void (*function)(void));

  // Returns why the interrupt happened. Two options are:
  // 0 - triggered because under reference voltage.
  // 1 - triggered because over reference voltage.
  uint8_t GetEvent() const { return get_lpcomp_event(); }

  // Use one-shot mode to read the battery voltage.
  int16_t MeasureBatteryVoltage();

  // This function converts the raw ADC reading into actual battery voltage.
  // The battery voltage can be converted later into battery percentage. This
  // could be done by measuring discharge curve.
  float ConvertBatteryVoltageToFloat(int16_t raw_adc_battery_reading);

 private:
  // Low power comparator definitions.
  enum {
    kLowPowerCompIrqPriority = 7  // lowest priority
  };

// Pin definitions.
#if PUCK_BOARD
  enum {
    kLowPowerCompPin = LPCOMP_PSEL_PSEL_AnalogInput6,  // AIN pin 6 (P0.30).
    kLowPowerCompAdcPin = SAADC_CH_PSELP_PSELP_AnalogInput6
  };
#endif

#if SLIM_BOARD
  enum {
    kLowPowerCompPin = LPCOMP_PSEL_PSEL_AnalogInput3,  // AIN pin 3 (P0.05).
    kLowPowerCompAdcPin = SAADC_CH_PSELP_PSELP_AnalogInput3
  };
#endif
};

#if SLIM_V2_BOARD
enum {
  // The low power comparator (LPCOMP) and analog-to-digital converter (ADC) are
  // connected to the same pin, but they are enumerated differently in hardware
  // abstraction layer (HAL), so
  // we need two separate definitions here.
  kLowPowerCompPin = LPCOMP_PSEL_PSEL_AnalogInput6,  // AIN pin 6 (P0.30).
  kLowPowerCompAdcPin = SAADC_CH_PSELP_PSELP_AnalogInput6
};
#endif

extern BatteryMonitor PuckBatteryMonitor;

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_BATTERY_MONITOR_H_
