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
//
//
// This driver provides a callback if the thermistor temperature is too high and
// it can measure the temperature on request. The goal is to prevent tactors
// from overheating. The driver is mostly written in HAL layer, but has few
// commands on register level, which are not available in HAL.
//
// A low power comparator (LPCOMP) hardware module is used to check if the
// voltage output of the thermistor is over a threshold (50 Celsius now). The
// threshold can be adjusted. LPCOMP module hardware is described here:
// https://infocenter.nordicsemi.com/index.jsp?topic=%2Fcom.nordic.infocenter.nrf52832.ps.v1.1%2Flpcomp.html
// To measure the analog battery voltage, SAADC was used.
//
// We use a 10K NTC thermistor (NTCG103JF103FT1), as described here:
// https://product.tdk.com/info/en/catalog/datasheets/503021/tpd_commercial_ntc-thermistor_ntcg_en.pdf
// B-constant is 3401-3450. Thermistor is connected with a 10K resistor to form
// a voltage divider.

#ifndef AUDIO_TO_TACTILE_SRC_TEMPERATURE_MONITOR_H_
#define AUDIO_TO_TACTILE_SRC_TEMPERATURE_MONITOR_H_

#include <cstdint>

#include "nrf_lpcomp.h"  // NOLINT(build/include)
#include "nrf_saadc.h"   // NOLINT(build/include)

namespace audio_tactile {

// Low power comparator definitions.
enum {
  LPCOMP_IRQ_PRIORITY = 7  // lowest priority
};

class TemperatureMonitor {
 public:
  TemperatureMonitor();

  // Configure the high temperature comparator interrupt.
  // Now it is set to trigger interrupt 50 Celsius.
  void StartMonitoringTemperature();

  // Stop the comparator and thus the temperature monitoring.
  void StopMonitoringTemperature();

  // This function is called when a high temperature is detected by comparator.
  void IrqHandler();

  // Allows the user to add high temperature warning in other parts of firmware.
  void OnOverheatingEventListener(void (*function)(void));

  // Returns why the interrupt happened. Two options are:
  // 0 - triggered because under reference voltage.
  // 1 - triggered because over reference voltage;
  uint8_t GetEvent() const { return event_; }

  // Use one-shot mode to read the thermistor voltage.
  int16_t TakeAdcSample();

  // This function converts the raw ADC reading into temperature (in Celsius)
  // The temperature is a few degrees higher than it should be. I get around 27
  // Celsius in room temperature. For accurate temperature, calibration with a
  // reference is required. However, here we use temperature only to detect if
  // tactors are overheating, so accuracy isn't critical.
  float ConvertAdcSampleToTemperature(int16_t raw_adc_battery_reading);

 private:
  // Callback for the interrupt.
  void (*callback_)(void);

  // Storing interrupt event. Event is info passed from the interrupt handler.
  uint8_t event_;
};

extern TemperatureMonitor SleeveTemperatureMonitor;

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_TEMPERATURE_MONITOR_H_
