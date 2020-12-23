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
// Driver for UI elements on the puck: navigation switch and 2 buttons.
//
// The driver does not use HAL or DRV so its self-contained. I use callback
// structure. The UI elements invoke callback function when they are pressed.
// Under the hood, the GPIO tasks and events module (GPIOE) is used. It is
// described here:
// https://infocenter.nordicsemi.com/index.jsp?topic=%2Fcom.nordic.infocenter.nrf52832.ps.v1.1%2Ftimer.html
// I use 5 out 8 GPIOE channels, each button has its own channel.
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

// Pin definitions.
enum {
  TACTILE_SW_1_PIN = 20,
  TACTILE_SW_2_PIN = 23,
  NAV_SW_CW_PIN = 43,
  NAV_SW_CCW_PIN = 45,
  NAV_SW_PRESS_PIN = 44
};

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

  // Configure all the button pins and initialize interrupts.
  // This function starts the lister (interrupt handler) as well.
  bool Initialize();

  // Stop the callbacks for buttons.
  bool End();

  // This function is called when a button touch is detected.
  void IrqHandler();

  // Allows the user to add a ui listener event in other parts of firmware.
  void OnUiEventListener(void (*function_)(void));

  // Returns which button was pressed. Options are:
  // 0 - Tactile button 1.
  // 1 - Tactile button 2.
  // 2 - Navigation switch clock-wise.
  // 3 - Navigation switch counter clock-wise.
  // 4 - Navigation switch press.
  // If two buttons are pressed within a timeout period, second press is not
  // passed to the event listener.
  uint8_t GetEvent() const { return event_; }

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

  // Storing button event.
  uint8_t event_;
};

extern Ui PuckUi;

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_UI_H_
