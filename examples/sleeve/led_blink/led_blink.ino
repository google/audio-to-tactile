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
// This is a simple sleeve app to blink an led.

const int  kLed  = 22;  // Led on pin P0.12

void setup() {
  pinMode(kLed, OUTPUT);
}

void loop() {
  digitalWrite(kLed, 1);
  delay(500);
  digitalWrite(kLed, 0);
  delay(500);
}