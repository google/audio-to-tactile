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

#include "pdm.h"  // NOLINT(build/include)

#include <cstring>

#include "nrf_gpio.h"  // NOLINT(build/include)

namespace audio_tactile {

void PdmMic::Initialize(uint16_t clock_pin, uint16_t data_pin) {
  // Set the buffer pointer for Easy DMA, thats where the PDM data goes.
  nrf_pdm_buffer_set(NRF_PDM, (uint32_t *)pdm_buffer_[0], kPdmDataSize);

  // Set the clock speed of the pdm module to 1 MHz. This is the clock pin rate,
  // essentially rate at which a bit is clocked out.
  nrf_pdm_clock_set(NRF_PDM, (nrf_pdm_freq_t)PDM_PDMCLKCTRL_FREQ_1000K);

  // Set the mode to mono (one channel) and bit on left rising edge.
  nrf_pdm_mode_set(NRF_PDM, (nrf_pdm_mode_t)NRF_PDM_MODE_MONO,
                   (nrf_pdm_edge_t)NRF_PDM_EDGE_LEFTRISING);

  // Set the left and right gains. It is an int from 0 to 80 (max).
  nrf_pdm_gain_set(NRF_PDM, kPdmDefaultGain, kPdmDefaultGain);

  // Clear the pins first.
  nrf_gpio_cfg_output(clock_pin);
  nrf_gpio_pin_clear(clock_pin);
  nrf_gpio_cfg_input(data_pin, NRF_GPIO_PIN_PULLDOWN);
  // Connect the pins to the mic.
  nrf_pdm_psel_connect(NRF_PDM, clock_pin, data_pin);

  // initialize and connect to interrupt handler.
  nrf_pdm_int_enable(
      NRF_PDM, NRF_PDM_INT_STARTED | NRF_PDM_INT_END | NRF_PDM_INT_STOPPED);
  NRFX_IRQ_PRIORITY_SET(PDM_IRQn, kPdmIrqPriority);
}

void PdmMic::Enable() {
  NRFX_IRQ_ENABLE(PDM_IRQn);
  nrf_pdm_enable(NRF_PDM);
  nrf_pdm_event_clear(NRF_PDM, NRF_PDM_EVENT_STARTED);
  nrf_pdm_task_trigger(NRF_PDM, NRF_PDM_TASK_START);
}

void PdmMic::Disable() {
  nrf_pdm_disable(NRF_PDM);
  NRFX_IRQ_DISABLE(PDM_IRQn);
}

void PdmMic::GetData(int16_t *destination_array) {
  memcpy(destination_array, pdm_buffer_[which_buffer_ready_],
         kPdmDataSize * sizeof(int16_t));
}

void PdmMic::SetMicGain(int16_t gain) { nrf_pdm_gain_set(NRF_PDM, gain, gain); }

PdmMic OnBoardMic;

void PdmMic::OnPdmDataReady(void (*function)(void)) { callback_ = function; }

void PdmMic::IrqHandler() {
  // Finished writing to Easy DMA buffer. The next cycle is automatically
  // started.
  if (nrf_pdm_event_check(NRF_PDM, NRF_PDM_EVENT_END)) {
    nrf_pdm_event_clear(NRF_PDM, NRF_PDM_EVENT_END);
    event_ = true;
  }

  // new data collection started, quickly set the new buffer here.
  if (nrf_pdm_event_check(NRF_PDM, NRF_PDM_EVENT_STARTED)) {
    nrf_pdm_event_clear(NRF_PDM, NRF_PDM_EVENT_STARTED);

    // PDM is naturally double buffered, so we need to take care of that.
    // See where the buffer pointer is. If its at first buffer, set next pointer
    // to second buffer and vise-versa.
    const int i =
        (nrf_pdm_buffer_get(NRF_PDM) == (uint32_t *)pdm_buffer_[0]) ? 1 : 0;
    nrf_pdm_buffer_set(NRF_PDM, (uint32_t *)pdm_buffer_[i], kPdmDataSize);
    which_buffer_ready_ = i;
  }

  // PDM module is stopped.
  if (nrf_pdm_event_check(NRF_PDM, NRF_PDM_EVENT_STOPPED)) {
    nrf_pdm_event_clear(NRF_PDM, NRF_PDM_EVENT_STOPPED);
    nrf_pdm_disable(NRF_PDM);
  }

  // Pass the event if new data is collected.
  if (callback_ && event_) {
    callback_();
    event_ = false;
  }
}

extern "C" {
void PDM_IRQHandler() { OnBoardMic.IrqHandler(); }
}

}  // namespace audio_tactile
