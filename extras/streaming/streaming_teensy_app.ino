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
// Teensy 4.0 app for streaming tactile data to the sleeve.
// This app takes the data from the USB data from the pc and sends it to sleeve.
// Syncronization is done between the Teensy and the sleeve so serial data to
// prevent gaps in playback. Teensy does not need to buffer the data as full
// speed usb is fast enough.

const int kNumChannels = 12;
const int kPwmSamples = 8;
const int kDataBytes = 128;
const int kHeaderBytes = 4;
byte inputArray[4096];
int serial_counter = 0;
unsigned long time_prev;
// The puck expects 128 data bytes and 4 header bytes in one serial packet.
byte received_tactile_frame[kHeaderBytes + kDataBytes];
volatile bool sequence_finished = false;
volatile bool new_data = false;

void setup() {
  SerialUSB.begin(1000000);
  Serial1.begin(1000000);
}

void loop() {
  if (SerialUSB.available()) {
    while (SerialUSB.available()) {
      // get the new byte:
      byte c = SerialUSB.read();
      inputArray[serial_counter] = c;
      serial_counter = serial_counter + 1;
    }

    new_data = true;

    // Get the timestamp to make sure there is no lag on USB port.
    // Should be around 4 ms for real-time performance.
    int time_stamp = (int)(time_prev - micros());
    SerialUSB.println(time_stamp);
    memcpy(received_tactile_frame, inputArray,
           (kNumChannels * kPwmSamples) + kHeaderBytes);

    serial_counter = 0;
    time_prev = micros();
  }  // end serial USB

  // send the data to the sleeve, and request new packet.
  if (sequence_finished && new_data) {
    Serial1.write(received_tactile_frame, 132);
    sequence_finished = false;
    new_data = false;
    // By sending buffer_copied to PC, new frame is sent.
    SerialUSB.println("buffer_copied");
  }
}

// Check for sync serial packet from the sleeve.
void serialEvent1() {
  byte c = Serial1.read();

  if (c == 23) {
    sequence_finished = true;
  }
}
 is done between the Teensy and the sleeve so serial data to
// prevent gaps in playback. Teensy does not need to buffer the data as full
// speed usb is fast enough.

const int kNumChannels = 12;
const int kPwmSamples = 8;
const int kDataBytes = 128;
const int kHeaderBytes = 4;
byte inputArray[4096];
int serial_counter = 0;
unsigned long time_prev;
// The sleeve expects 128 data bytes and 4 header bytes in one serial packet.
byte received_tactile_frame[kHeaderBytes + kDataBytes];
volatile bool sequence_finished = false;
volatile bool new_data = false;

void setup() {
  SerialUSB.begin(1000000);
  Serial1.begin(1000000);
}

void loop() {
  if (SerialUSB.available()) {
    while (SerialUSB.available()) {
      // get the new byte:
      byte c = SerialUSB.read();
      inputArray[serial_counter] = c;
      serial_counter = serial_counter + 1;
    }

    new_data = true;

    // Get the timestamp to make sure there is no lag on USB port.
    // Should be around 4 ms for real-time performance.
    int time_stamp = (int)(time_prev - micros());
    SerialUSB.println(time_stamp);
    memcpy(received_tactile_frame, inputArray,
           (kNumChannels * kPwmSamples) + kHeaderBytes);

    serial_counter = 0;
    time_prev = micros();
  }  // end serial USB

  // send the data to the sleeve, and request new packet.
  if (sequence_finished && new_data) {
    Serial1.write(received_tactile_frame, 132);
    sequence_finished = false;
    new_data = false;
    // By sending buffer_copied to PC, new frame is sent.
    SerialUSB.println("buffer_copied");
  }
}

// Check for sync serial packet from the sleeve.
void serialEvent1() {
  byte c = Serial1.read();

  if (c == 23) {
    sequence_finished = true;
  }
}
