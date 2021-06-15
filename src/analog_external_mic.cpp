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

#include "analog_external_mic.h"  // NOLINT(build/include)

namespace audio_tactile {

AnalogMic::AnalogMic() {}

void AnalogMic::Initialize() {
  // Found discussion here (warning: the code has problems):
  // https://devzone.nordicsemi.com/f/nordic-q-a/57057/not-getting-data-from-adc-in-continuous-mode

  // Configure microphone shutdown pin.
  nrf_gpio_cfg_output(kMicShutDownPin);

  // Turn on the microphone amplifier.
  nrf_gpio_pin_write(kMicShutDownPin, 1);

  nrf_saadc_resolution_set(NRF_SAADC, NRF_SAADC_RESOLUTION_12BIT);

  // Configure the ADC channel.
  nrf_saadc_channel_config_t channel_config = {
      .resistor_p = NRF_SAADC_RESISTOR_DISABLED,
      .resistor_n = NRF_SAADC_RESISTOR_DISABLED,
      .gain = NRF_SAADC_GAIN1_6,
      .reference = NRF_SAADC_REFERENCE_INTERNAL,
      .acq_time = NRF_SAADC_ACQTIME_3US,  // The output impedance is about 10k,
                                          // so 3us is ok.
      .mode = NRF_SAADC_MODE_DIFFERENTIAL,
      .burst = NRF_SAADC_BURST_DISABLED};

  nrf_saadc_channel_init(NRF_SAADC, 0, &channel_config);

  // Connect to the pins.
  nrf_saadc_channel_pos_input_set(NRF_SAADC, 0, (nrf_saadc_input_t)kMicAdcPin);

  // Set SAADC to continuous sampling using the internal timer SAADC timer.
  // Sample Rate is 16 MHz / CC register.
  // Setting CC to 1024 gives 15625 Hz sampling rate.
  nrf_saadc_continuous_mode_enable(NRF_SAADC, 1024);

  // Enable SAADC global interrupt.
  NVIC_DisableIRQ(SAADC_IRQn);
  NVIC_ClearPendingIRQ(SAADC_IRQn);
  NVIC_SetPriority(SAADC_IRQn, kSaadcIrqPriority);
  NVIC_EnableIRQ(SAADC_IRQn);

  // Enable specific interrupts. Don't care about other ones.
  nrf_saadc_int_enable(NRF_SAADC, NRF_SAADC_INT_END);
  nrf_saadc_int_enable(NRF_SAADC, NRF_SAADC_INT_STOPPED);

  // Set the buffer for EASY DMA transfer.
  nrf_saadc_buffer_init(NRF_SAADC, adc_buffer_, kAdcDataSize);

  // Enable SAADC.
  nrf_saadc_enable(NRF_SAADC);

  // Start SAADC.
  nrf_saadc_task_trigger(NRF_SAADC, NRF_SAADC_TASK_START);

  // Wait until its ready to go.
  while (!NRF_SAADC->EVENTS_STARTED) {
  }
  nrf_saadc_event_clear(NRF_SAADC, NRF_SAADC_EVENT_STARTED);

  // Start sampling, from now SAADC is handled by the interrupt routine.
  nrf_saadc_task_trigger(NRF_SAADC, NRF_SAADC_TASK_SAMPLE);
}

void AnalogMic::Disable() {
  nrf_saadc_disable(NRF_SAADC);
  NVIC_DisableIRQ(SAADC_IRQn);
  nrf_gpio_pin_write(kMicShutDownPin, 0);
}

void AnalogMic::Enable() {
  nrf_gpio_pin_write(kMicShutDownPin, 1);
  nrf_saadc_enable(NRF_SAADC);
  NVIC_EnableIRQ(SAADC_IRQn);
}

void AnalogMic::GetData(int16_t* destination_array) {
  memcpy(destination_array, adc_buffer_, kAdcDataSize * sizeof(int16_t));
}

AnalogMic ExternalAnalogMic;

void AnalogMic::OnAdcDataReady(void (*function)(void)) { callback_ = function; }

// Interrupt handler for the low power comparator rederects to IrqHandler().
extern "C" {
void SAADC_IRQHandler() { ExternalAnalogMic.IrqHandler(); }
}

void AnalogMic::IrqHandler() {
  bool event;
  // Triggered when data points are collected,
  if (nrf_saadc_event_check(NRF_SAADC, NRF_SAADC_EVENT_END)) {
    nrf_saadc_event_clear(NRF_SAADC, NRF_SAADC_EVENT_END);
    nrf_saadc_buffer_init(NRF_SAADC, adc_buffer_, kAdcDataSize);
    nrf_saadc_task_trigger(NRF_SAADC, NRF_SAADC_TASK_START);
  }

  // Triggered when data is trasfered to RAM with Easy DMA.
  if (nrf_saadc_event_check(NRF_SAADC, NRF_SAADC_EVENT_DONE)) {
    nrf_saadc_event_clear(NRF_SAADC, NRF_SAADC_EVENT_DONE);
    event = true;
  }

  // Triggered after ADC is stopped.
  if (nrf_saadc_event_check(NRF_SAADC, NRF_SAADC_EVENT_STOPPED)) {
    nrf_saadc_event_clear(NRF_SAADC, NRF_SAADC_EVENT_STOPPED);
  }

  // Pass the event if new data is collected.
  if (callback_ && event) {
    callback_();
  }
}

}  // namespace audio_tactile
