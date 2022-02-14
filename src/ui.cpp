// Copyright 2020-2022 Google LLC
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

#include "ui.h"  // NOLINT(build/include)

namespace audio_tactile {

bool static first_click = true;

Ui::Ui() {}

bool Ui::Initialize(int switch_pin) {
  // Found an example:
  // https://devzone.nordicsemi.com/f/nordic-q-a/5358/how-do-i-use-an-external-interrupt
  // low-lewel GPIOTE (GPIO tasks and events) description is here:
  // https://infocenter.nordicsemi.com/index.jsp?topic=%2Fcom.nordic.infocenter.nrf52832.ps.v1.1%2Fgpiote.html

  // Set pin as inputs with an internal pullup.
  nrf_gpio_cfg_input(switch_pin, NRF_GPIO_PIN_PULLUP);

  // Clear previous and enable interrupts.
  NVIC_DisableIRQ(GPIOTE_IRQn);
  NVIC_ClearPendingIRQ(GPIOTE_IRQn);
  NVIC_SetPriority(GPIOTE_IRQn, GPIOTE_IRQ_PRIORITY);
  NVIC_EnableIRQ(GPIOTE_IRQn);

  constexpr uint32_t kSharedConfig =
      // Configure channel in event mode (tracking if something happens on a
      // pin).
      ((GPIOTE_CONFIG_MODE_Event << GPIOTE_CONFIG_MODE_Pos)
       // Clear the polarity (trigger mode).
       & ~(GPIOTE_CONFIG_PORT_PIN_Msk | GPIOTE_CONFIG_POLARITY_Msk))
      // Set the trigger mode.
      | ((GPIOTE_CONFIG_POLARITY_HiToLo << GPIOTE_CONFIG_POLARITY_Pos) &
         GPIOTE_CONFIG_POLARITY_Msk);

  // Apply the configuration to the channel.
  NRF_GPIOTE->CONFIG[0] =
      kSharedConfig |
      ((switch_pin << GPIOTE_CONFIG_PSEL_Pos) & GPIOTE_CONFIG_PORT_PIN_Msk);

  // Enable the interrupt for the channel.
  NRF_GPIOTE->INTENSET = GPIOTE_INTENSET_IN0_Set << GPIOTE_INTENSET_IN0_Pos;

  // Initialize timer for debouncing.
  TimerInit();

  return 0;
}

void Ui::OnUiEventListener(void (*function_)(void)) { callback_ = function_; }

Ui DeviceUi;

void Ui::IrqHandler() {
  NVIC_DisableIRQ(GPIOTE_IRQn);

  if (NRF_GPIOTE->EVENTS_IN[0] == 1) {
    NRF_GPIOTE->EVENTS_IN[0] = 0;
  }

  // Make sure to debounce before triggering a callback.
  if (callback_ && first_click) {
    callback_();
  }

  // Start the timer after the first interrupt.
  if (first_click) {
    TimerStart();
    first_click = false;
  }
  NVIC_EnableIRQ(GPIOTE_IRQn);
}

extern "C" {
void GPIOTE_IRQHandler() { DeviceUi.IrqHandler(); }
}

// Timer interrupt handler called when timer reaches DEBOUNCE_TIMEOUT_US.
extern "C" {
void TIMER3_IRQHandler(void) {
  // Disable timeout when this timer runs out.
  if (NRF_TIMER3->EVENTS_COMPARE[0]) {
    NRF_TIMER3->EVENTS_COMPARE[0] = 0;
    first_click = true;

    // Stop the timer.
    NRF_TIMER3->TASKS_CLEAR = 1;
    NRF_TIMER3->TASKS_STOP = 1;
  }
}
}

void Ui::TimerInit() {
  // Set the registers accrording to the datasheet:
  // https://infocenter.nordicsemi.com/index.jsp?topic=%2Fcom.nordic.infocenter.nrf52832.ps.v1.1%2Ftimer.html
  // Set the timer 3 to 24-bitmode.
  NRF_TIMER3->BITMODE = TIMER_BITMODE_BITMODE_24Bit
                        << TIMER_BITMODE_BITMODE_Pos;

  // Set the prescale for an interval of 1 us (16M / (2^4)).
  NRF_TIMER3->PRESCALER = 4;

  // Set interrupt after x microseconds.
  NRF_TIMER3->CC[0] = DEBOUNCE_TIMEOUT_US;

  // Make sure the timer clears after reaching CC[0].
  NRF_TIMER3->SHORTS = TIMER_SHORTS_COMPARE0_CLEAR_Msk;

  // Trigger the interrupt when reaching CC[0].
  NRF_TIMER3->INTENSET = TIMER_INTENSET_COMPARE0_Msk;

  // Set a low IRQ priority and enable interrupts for the timer module.
  NVIC_SetPriority(TIMER3_IRQn, TIMEOUT_IRQ_PRIORITY);
  NVIC_EnableIRQ(TIMER3_IRQn);
}

void Ui::TimerStart() {
  NRF_TIMER3->TASKS_CLEAR = 1;
  NRF_TIMER3->TASKS_START = 1;
}

void Ui::TimerStop() {
  NRF_TIMER3->TASKS_CLEAR = 1;
  NRF_TIMER3->TASKS_STOP = 1;
}

uint32_t Ui::TimerGetValueUsec() {
  NRF_TIMER3->TASKS_CAPTURE[1] = 1;
  return (uint32_t)NRF_TIMER3->CC[1];
}

}  // namespace audio_tactile
