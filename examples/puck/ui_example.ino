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
// This is an example of using buttons on the puck board.
// The example uses a callback in case any button is pressed.
// Debouncing is done in the library.
// GetEvent returns which button is pressed.

#include "ui.h"

using namespace audio_tactile;

void setup() {
  pinMode(27, OUTPUT); //LED on P.19
  PuckUi.Initialize();
  PuckUi.OnUiEventListener(touch_event);
}

void loop() {
}

void touch_event() {
  digitalToggle(27);
  Serial.println(PuckUi.GetEvent());
}