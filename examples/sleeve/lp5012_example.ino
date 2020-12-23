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
// This is a simple Arduino example of using Lp5012 LED chip on the sleeve
// board. The program will cycle leds one by one and then all at once.

#include "lp5012.h"
#include "two_wire.h"

using namespace audio_tactile;

void setup() {
  LedArray.Initialize();
  LedArray.CycleLedsOneByOne(3);
  LedArray.CycleAllLeds(3);
}

void loop() {}
