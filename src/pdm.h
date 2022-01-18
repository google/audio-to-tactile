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
// Driver for the on-board CMM-4030DT-26354 PDM microphone from CUI devices.
//
// Datasheet is here:
// https://www.cuidevices.com/product/resource/cmm-4030dt-26354-tr.pdf
// The pdm driver was build using the nordic HAL interface. HAL description:
// https://infocenter.nordicsemi.com/index.jsp?topic=%2Fcom.nordic.infocenter.sdk5.v12.0.0%2Fgroup__nrf__pdm__hal.html
// Microphone sampling rate is 16 kHz.

#ifndef AUDIO_TO_TACTILE_SRC_PDM_H_
#define AUDIO_TO_TACTILE_SRC_PDM_H_

#include <stdint.h>

#include "board_defs.h"  // NOLINT(build/include)
#include "nrf_pdm.h"  // NOLINT(build/include)

namespace audio_tactile {

// Buffer constants.
const int kPdmDataSize = 64;
const int kNumberPdmBuffers = 2;

class PdmMic {
 public:
  // Pdm buffer constants.
  enum {
    kPdmDataSize = 64,
    kNumberPdmBuffers = 2,
  };

  // Initialize the PDM microphone.
  void Initialize(uint16_t clock_pin, uint16_t data_pin);

  // Start audio data collection.
  void Enable();

  // Get the mic data array. The data is updated automatically in IRQ.
  void GetData(int16_t* destination_array);

  // Disable the PDM mic.
  void Disable();

  // Set the microphone gain. Value from 0 (min) to 80 (max).
  // Default gain is 80 (max).
  void SetMicGain(int16_t gain);

  // Set callback function to be called when there is interrupt from PDM module.
  void IrqHandler();

  // This function is called when new data is ready. Good for real-time
  // processing.
  void OnPdmDataReady(void (*function)(void));

 private:
  // Callback for the interrupt.
  void (*callback_)(void);

  // Two PDM buffers are required, as they swapped after filling.
  int16_t pdm_buffer_[kNumberPdmBuffers][kPdmDataSize];

  // Storing interrupt event.
  bool event_;

  // Which buffer is currently ready to read.
  uint8_t which_buffer_ready_;

  // PDM hardware constants.
  enum {
    kPdmIrqPriority = 6,
    kPdmDefaultGain = 80  // set to max
  };
};

extern PdmMic OnBoardMic;

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_PDM_H_
