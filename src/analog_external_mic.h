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
// Arduino-compatible driver library for an external analog microphone.
//
// HAL (Hardware abstraction layer) functions are used, but Nordic's driver is
// not. We use the MAX9814 amplifier with the output gain of 60dB. The datasheet
// is here: https://datasheets.maximintegrated.com/en/ds/MAX9814.pdf
//
// For the ADC, we use SAADC module (Successive Approximation
// Analog-to-Digital-Converter) The ADC is running in a continuous mode at 15625
// Hz sample rate. In a continuous mode, the sampling does not require CPU
// intervention at every sample, we use an internal ADC timer for timing. It
// does not need any external timers, but on the other hand, only one channel
// can work. This is not an issue here, as we only have one analog microphone.
// Interrupts are used to take care of the data and setup the EASY DMA. It is
// unintuitive how the ADC interrupts function, Task start has to be triggered
// after the data is collected, even in the continuous mode. Also, EVENT_DONE
// and EVENT_END are confusing. EVENT_END is triggered when data is collected,
// and EVENT_DONE when data is transferred to RAM buffer. Easy DMA buffer and
// task trigger has to be initiated on EVENT_END, otherwise it leads to a
// hardfault. The ADC is documented here:
// https://infocenter.nordicsemi.com/index.jsp?topic=%2Fcom.nordic.infocenter.nrf52832.ps.v1.1%2Fsaadc.html

#ifndef AUDIO_TO_TACTILE_SRC_ANALOG_EXTERNAL_MIC_H_
#define AUDIO_TO_TACTILE_SRC_ANALOG_EXTERNAL_MIC_H_

#include <stdint.h>
#include <string.h>

#include "nrf_gpio.h"   // NOLINT(build/include)
#include "nrf_saadc.h"  // NOLINT(build/include)

namespace audio_tactile {

// Hardware constants.
enum {
  kSaadcIrqPriority = 7,  // lowest priority
  kMicShutDownPin = 46    // On P1.14, which maps to 32 + 14 = 46
};

const uint16_t kAdcDataSize = 64;

class AnalogMic {
 public:
  AnalogMic();

  // Configure the ADC.
  // This function starts the listener (interrupt handler) as well.
  void Initialize();

  // Stop the callbacks, disables the ADC.
  void Disable();

  // Start the callbacks, enable the ADC.
  void Enable();

  // This function is called when buffer is filled with new data.
  void IrqHandler();

  // Returns the latest data in the buffer.
  void GetData(int16_t* destination_array);

  // This function is called when new data is ready. Good for real-time
  // processing.
  void OnAdcDataReady(void (*function)(void));

 private:
  // Callback for the interrupt.
  void (*callback_)(void);

  // Buffer for the collected analog data.
  nrf_saadc_value_t adc_buffer_[kAdcDataSize];
};

extern AnalogMic ExternalAnalogMic;

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_ANALOG_EXTERNAL_MIC_H_
