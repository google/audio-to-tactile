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

#include "temperature_monitor.h"  // NOLINT(build/include)

#include <math.h>

namespace audio_tactile {

TemperatureMonitor::TemperatureMonitor() {}

void TemperatureMonitor::StartMonitoringTemperature() {
  // Enable interrupt on LPCOMP CROSS event.
  NRF_LPCOMP->INTENSET = LPCOMP_INTENSET_CROSS_Msk;

  // Clear previous and enable interrupts. Set priority.
  NVIC_DisableIRQ(LPCOMP_IRQn);
  NVIC_ClearPendingIRQ(LPCOMP_IRQn);
  NVIC_SetPriority(LPCOMP_IRQn, LPCOMP_IRQ_PRIORITY);
  NVIC_EnableIRQ(LPCOMP_IRQn);

  // We want to trigger overheating warning interrupt when temperature goes over
  // 50 Celsius. 50 Celsius raw ADC value is around 2600 or 2.3 V. While, at 27
  // Celsius raw ADC value is around 2000 or 1.7 V. The trigger can only be one
  // of the 16 values. I set the trigger voltage to 1/16 of Vdd = 3.3 V *
  // (11/16) = 2.27 V.
  NRF_LPCOMP->REFSEL |=
      (LPCOMP_REFSEL_REFSEL_Ref11_16Vdd << LPCOMP_REFSEL_REFSEL_Pos);

  // Set reference input source to analog in (AIN) pin 1.
  nrf_lpcomp_input_select(NRF_LPCOMP, NRF_LPCOMP_INPUT_1);

  // Enable and start the low power comparator.
  nrf_lpcomp_enable(NRF_LPCOMP);
  nrf_lpcomp_task_trigger(NRF_LPCOMP, NRF_LPCOMP_TASK_START);
}

void TemperatureMonitor::StopMonitoringTemperature() {
  // Disable the low power comparator.
  nrf_lpcomp_disable(NRF_LPCOMP);
  // Disable the interrupt handler.
  NVIC_DisableIRQ(LPCOMP_IRQn);
  NVIC_ClearPendingIRQ(LPCOMP_IRQn);
}

int16_t TemperatureMonitor::TakeAdcSample() {
  // This code was inspired by the Arduino AnalogRead library.
  // https://github.com/adafruit/Adafruit_nRF52_Arduino/blob/master/cores/nRF5/wiring_analog_nRF52.c
  static int16_t value = 0;

  nrf_saadc_enable(NRF_SAADC);
  nrf_saadc_resolution_set(NRF_SAADC, NRF_SAADC_RESOLUTION_12BIT);

  // Use ADC channel 0. Configure the channel here.
  // Set the acquisition time to 3 us, since output impedance is about 10 kOhms.
  // Select internal reference (0.6V), which gives us 3.6 V range.
  nrf_saadc_channel_config_t channel_config = {
      .resistor_p = NRF_SAADC_RESISTOR_DISABLED,
      .resistor_n = NRF_SAADC_RESISTOR_DISABLED,
      .gain = NRF_SAADC_GAIN1_6,
      .reference = NRF_SAADC_REFERENCE_INTERNAL,
      .acq_time = NRF_SAADC_ACQTIME_3US,
      .mode = NRF_SAADC_MODE_SINGLE_ENDED,
      .burst = NRF_SAADC_BURST_DISABLED};

  nrf_saadc_channel_input_set(NRF_SAADC, 0, NRF_SAADC_INPUT_AIN1,
                              NRF_SAADC_INPUT_DISABLED);

  nrf_saadc_channel_init(NRF_SAADC, 0, &channel_config);

  // Initiate EasyDMA buffer. Will give a hardfault if we start adc without a
  // buffer.
  nrf_saadc_buffer_init(NRF_SAADC, &value, 1);

  // Trigger immediate ADC sampling.
  nrf_saadc_task_trigger(NRF_SAADC, NRF_SAADC_TASK_START);

  while (!NRF_SAADC->EVENTS_STARTED) {
  }

  nrf_saadc_event_clear(NRF_SAADC, NRF_SAADC_EVENT_STARTED);
  nrf_saadc_task_trigger(NRF_SAADC, NRF_SAADC_TASK_SAMPLE);

  while (!NRF_SAADC->EVENTS_END) {
  }

  nrf_saadc_event_clear(NRF_SAADC, NRF_SAADC_EVENT_END);
  nrf_saadc_task_trigger(NRF_SAADC, NRF_SAADC_TASK_STOP);

  while (!NRF_SAADC->EVENTS_STOPPED) {
  }

  nrf_saadc_event_clear(NRF_SAADC, NRF_SAADC_EVENT_STOPPED);

  // Sometimes ADC values can be negative, just make them 0.
  if (value < 0) {
    value = 0;
  }

  // Disable the ADC.
  nrf_saadc_disable(NRF_SAADC);

  return value;
}

float TemperatureMonitor::ConvertAdcSampleToTemperature(
    int16_t raw_adc_battery_reading) {
  float voltage;
  float r_therm;
  float temperature;

  // There are three conversions that need to be performed:
  // 1) Adc sample -> voltage. 2) Voltage -> resistance 3) Resistance ->
  // temperature. I used this reference tutorial:
  // https://www.jameco.com/Jameco/workshop/techtip/temperature-measurement-ntc-thermistors.html#:~:text=The%20variable%20T%20is%20the,thermistor%20resistance%20at%20temperature%20T0.
  // The conversion is done using a simplified Steinhart-Hart equation.

  // First convert ADC reading to voltage.
  // The ADC gain is set to 1/6.
  const float kGainADC = 1.0f / 6.0f;

  // This is the internal reference voltage used by ADC.
  const float kInternalRefVoltage = 0.6f;

  // The resolution of the ADC is 12 bits (2^12).
  const float kBitsResolution = 4096.0f;

  // Thermistor supply voltage.
  const float kSupplyVoltage = 3.3f;

  // The input range is reference voltage divided by gain, according to the
  // SAADC datasheet:
  // https://infocenter.nordicsemi.com/index.jsp?topic=%2Fcom.nordic.infocenter.nrf52832.ps.v1.1%2Fsaadc.html
  const float kInputRange = (kInternalRefVoltage) / (kGainADC);

  // Bit to volts conversion factor.
  const float kBitsToVolts = kInputRange / kBitsResolution;

  // Convert ADC reading to voltage.
  voltage = kBitsToVolts * raw_adc_battery_reading;

  // Now, convert the voltage to resistance

  // The resistor in the voltage divider has a value of 10K.
  const float kR2 = 10000.0f;

  r_therm = kR2 * ((kSupplyVoltage / voltage) - 1);

  // Convert resistance to temperature using simplified Steinhart-Hart equation.

  // Thermistor resistance at 25 Celsius (from datasheet)
  const float kRo = 10000.0f;

  // Thermistor nominal temperature is 25 Celsius or 298.15 Kelvin.
  const float kTo = 298.15f;

  // B-constant from the datasheet.
  const float kB_constant = 3450.0f;

  // Plug into simplified Steinhart-Hart equation.
  temperature = 1.0f / kTo + 1.0f / kB_constant * log(r_therm / kRo);
  temperature = 1 / temperature;
  temperature = temperature - 273.15f;  // Kelvin -> Celsius.

  return temperature;
}

TemperatureMonitor SleeveTemperatureMonitor;

void TemperatureMonitor::OnOverheatingEventListener(void (*function)(void)) {
  on_lpcomp_trigger(function);
}

}  // namespace audio_tactile
