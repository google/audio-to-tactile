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
//
//
// Driver for UI element on the slim v2 board: tactile button.
//
// The driver does not use HAL or DRV so its self-contained. I use callback
// structure. The UI elements invoke callback function when they are pressed.
// Under the hood, the GPIO tasks and events module (GPIOE) is used. It is
// described here:
// https://infocenter.nordicsemi.com/index.jsp?topic=%2Fcom.nordic.infocenter.nrf52832.ps.v1.1%2Ftimer.html
//
// To debounce the buttons, I use a timer, after a first click timer starts and
// while running makes the callback not respond for a certain time. I use the
// TIMER3 module on the nRF, if this timer is used by something else, other
// times could be used. The timer is described here:
// https://infocenter.nordicsemi.com/index.jsp?topic=%2Fcom.nordic.infocenter.nrf52832.ps.v1.1%2Ftimer.html

#ifndef AUDIO_TO_TACTILE_SRC_UI_H_
#define AUDIO_TO_TACTILE_SRC_UI_H_

#include <stdint.h>

#include "nrf_gpio.h"    // NOLINT(build/include)
#include "nrf_gpiote.h"  // NOLINT(build/include)

namespace audio_tactile {

// GPIO interrupt constants.
enum {
  GPIOTE_IRQ_PRIORITY = 6,
};

// TIMER constants.
enum {
  DEBOUNCE_TIMEOUT_US = 150000,  // 150 ms (in usec)
  TIMEOUT_IRQ_PRIORITY = 7       // lowest priority
};

class Ui {
 public:
  Ui();

  // Configure the button pin and initialize interrupts.
  // This function starts the lister (interrupt handler) as well.
  bool Initialize(int switch_pin);

  // Stop the callbacks for buttons.
  bool End();

  // This function is called when a button touch is detected.
  void IrqHandler();

  // Allows the user to add a ui listener event in other parts of firmware.
  void OnUiEventListener(void (*function_)(void));

 private:
  // Callback for the interrupt.
  void (*callback_)(void);

  // Initialize the timer.
  void TimerInit();

  // Clear and start the timer.
  void TimerStart();

  // Stop the timer.
  void TimerStop();

  // Return the current timer count.
  uint32_t TimerGetValueUsec();
};

extern Ui DeviceUi;

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_UI_H_
