// Copyright 2021 Google LLC
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
// Library for LIS3DH low-power accelerometer.
// Datasheet can be found here:
// https://www.st.com/resource/en/datasheet/lis3dh.pdf
//
// Other impelementations from Adafruit and Sparkfun:
// https://github.com/adafruit/Adafruit_LIS3DH
// https://github.com/sparkfun/SparkFun_LIS3DH_Arduino_Library
//
// The orientation of the accelerometer is as following (if worn on the left
// wrist with the electronics box facing the top):
// Right-handed coordinate system is used.
// - Positive X-axis oriented away from the hand, along the arm, towards the
// body. In anatomical terms, positive X-axis is the "proximal" direction.
// - Positive Y-axis oriented perpendicular to the hand and wrist. The positive
// axis point towards "medial" directions, which means towards the midline of
// the body.
// - Positive Z-axis points away from the top of the wrist in the normal
// direction. In the anatomical terms, it is "superficial" direction.
//
// A picture of the axis is shown in
// audio_to_tatile/extras/doc/axis_accelerometer_left_hand.jpg
//
// A simple graphic is shown below (Looking at the inner side of left hand,
// "left lateral direction ")
//          ^ +Z
//   +X     |
//   <--- device
// elbow  wrist  hand
//
// +Y is not shown as it points away from the screen.

#ifndef THIRD_PARTY_AUDIO_TO_TACTILE_SRC_ACCELEROMETER_LIS3DH_H_
#define THIRD_PARTY_AUDIO_TO_TACTILE_SRC_ACCELEROMETER_LIS3DH_H_

namespace audio_tactile {

class Lis3dh {
 public:
  Lis3dh();

  // Initialize the accelerometer and I2C bus.
  bool Initialize();

  // Enable accelerometer, wake up from sleep.
  void Enable();

  // Disable acceleromter, put to sleep.
  void Disable();

  // Read the XYZ acceleration data.
  // The data is returned as a array of three int16_t values corresponsing to X,
  // Y and Z axis respectively.
  const int16_t* ReadXyzAccelerationRaw();

  // Read the XYZ acceleration data, converted to gravity.
  // Currently the gravity is set to (-2 to +2 Gs).
  const float* ReadXyzAccelerationFloat();

 private:
  // Register map.
  enum {
    STATUS_REG_AUX = 0x07,
    OUT_ADC1_L = 0x08,
    OUT_ADC1_H = 0x09,
    OUT_ADC2_L = 0x0A,
    OUT_ADC2_H = 0x0B,
    OUT_ADC3_L = 0x0C,
    OUT_ADC3_H = 0x0D,
    WHO_AM_I = 0x0F,
    CTRL_REGO = 0x1E,
    TEMP_CFG_REG = 0x1F,
    CTRL_REG1 = 0x20,
    CTRL_REG2 = 0x21,
    CTRL_REG3 = 0x22,
    CTRL_REG4 = 0x23,
    CTRL_REG5 = 0x24,
    CTRL_REG6 = 0x25,
    STATUS_REG = 0x27,
    OUT_X_L = 0x28,
    OUT_X_H = 0x29,
    OUT_Y_L = 0x2A,
    OUT_Y_H = 0x2B,
    OUT_Z_L = 0x2C,
    OUT_Z_H = 0x2D,
    FIFO_CTRL_REG = 0x2E,
    FIFO_SRC_REG = 0x2F,
    INT1_CFG = 0x30,
    INT1_SRC = 0x31,
    INT1_THIS = 0x32,
    INT1_DURATION = 0x33,
    INT2_CFG = 0x34,
    INT2_SRC = 0x35,
    INT2_THIS = 0x36,
    INT2_DURATION = 0x37,
    CLICK_CFG = 0x38,
    CLICK_SRC = 0x39,
    CLICK_THIS = 0x3A,
    TIME_LIMIT = 0x3B,
    TIME_LATENCY = 0x3C,
    TIME_WINDOW = 0x3D,
    ACT_THIS = 0x3E,
    ACT_DUR = 0x3F,
  };

  // Hardware constants.
  enum {
    kSclPin = 25,
    kSdaPin = 24,
    kLis3dhAddress = 0x18,
    kWhoAmIResponse = 0x33,  // 0b00110011
  };
};

extern Lis3dh Accelerometer;

}  // namespace audio_tactile

#endif  // THIRD_PARTY_AUDIO_TO_TACTILE_SRC_ACCELEROMETER_LIS3DH_H_
