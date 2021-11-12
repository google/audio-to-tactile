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
// App for streaming tactile data to the slim board.
// This app takes the data from the USB from the pc and sends it to
// actuators. Full 12-channel data is expected at 16 bit resolution and 2kHz.
//
// The slim board buffers kNumBufferedFrames of pwm data to avoid USB timing
// issues. Double buffer is used to allow playback without gaps. Overall the
// timing is as follows:
// 1) Serial USB buffer is filled with kNumBufferedFrames
// 2) When full serial buffer is copied into pending playback buffer.
// 3) When PWM buffer of kNumBufferedFrames is done playing, it is filled from
// pending playback buffer.
// 4) When PWM buffer played kWhenToSendFrame frames, new USB
// data is requested.
//
// Note that PWM is always playing something, so if the buffer is not
// replenished, it will replay old data.
// To silence the tactors a buffer of zeros should be send.

#include "board_defs.h"
#include "pwm_sleeve.h"
#include "serial_com.h"

using namespace audio_tactile;

constexpr int kNumChannels = 12;
constexpr int kPwmSamples = 8;
constexpr int kDataBytes = kNumChannels * kPwmSamples;
constexpr int kHeaderBytes = 4;
// Number of buffered pwm frames.
constexpr int kNumBufferedFrames = 8;
// The new data is send after playing kWhenToSendFrame frames.
// There should enough time to fill the buffer to allow continous playback
// On MacOS, each frame takes ~2 ms to transmit and ~4 ms to play.
// To minimize latency kNumBufferedFrames can be decreased, depending on the USB
// driver.
// TODO: test on Linux and Windows OS.
constexpr int kWhenToSendFrame = 3;
constexpr int kSerialBufferSize = 2048;
constexpr int kPendingPwmBufferSize =
    kNumBufferedFrames * (kDataBytes + kHeaderBytes);

static uint8_t g_serial_input_buffer[kSerialBufferSize];
// Make sure that the serial buffer is large enough to hold kNumBufferedFrames.
static_assert(sizeof(g_serial_input_buffer) >=
                  ((kDataBytes + kHeaderBytes) * kNumBufferedFrames),
              "g_serial_input_buffer is not large enough");

static uint8_t g_pending_playback_pwm_buffer[kPendingPwmBufferSize];
static uint8_t g_playback_pwm_buffer[kPendingPwmBufferSize];
static int g_serial_counter = 0;
static uint8_t g_received_tactile_frame[kDataBytes];
volatile bool g_sequence_finished = false;
volatile uint8_t g_which_pwm_module_triggered;
static int g_frame_counter = 0;

void FlashLeds();
void OnPwmSequenceEnd();

void setup() {
  Serial.begin(1000000);
  nrf_gpio_cfg_output(kLedPinBlue);
  nrf_gpio_cfg_output(kLedPinGreen);

  // Initialize PWM.
  SleeveTactors.OnSequenceEnd(OnPwmSequenceEnd);
  SleeveTactors.Initialize();
  // Warning: issue only in Arduino. When using StartPlayback() it crashes.
  // Looks like NRF_PWM0 module is automatically triggered, and triggering it
  // again here crashes ISR. Temporary fix is to only use nrf_pwm_task_trigger
  // for NRF_PWM1 and NRF_PWM2. To fix might need a nRF52 driver update.
  // nrf_pwm_task_trigger(NRF_PWM0, NRF_PWM_TASK_SEQSTART0);
  nrf_pwm_task_trigger(NRF_PWM1, NRF_PWM_TASK_SEQSTART0);
  nrf_pwm_task_trigger(NRF_PWM2, NRF_PWM_TASK_SEQSTART0);

  FlashLeds();
}

void loop() {
  if (Serial.available()) {
    // Save the received bytes into an array.
    while (Serial.available()) {
      // Get the new byte:
      uint8_t c = Serial.read();
      // Exit execution on a serial buffer overrun.
      if (g_serial_counter >= kSerialBufferSize) {
        Serial.println("Buffer overrun in serial communication");
        // Force sleep.
        for (;;) {
          delay(1000);
        }
      }
      g_serial_input_buffer[g_serial_counter] = c;
      ++g_serial_counter;
      nrf_gpio_pin_toggle(kLedPinBlue);
    }

    Serial.println(g_serial_counter);

    // Copy the USB data into temp pwm buffer, and swap it once previous buffer
    // is finished playing.
    if (g_serial_counter ==
        ((kDataBytes + kHeaderBytes) * kNumBufferedFrames)) {
      memcpy(g_pending_playback_pwm_buffer, g_serial_input_buffer,
             g_serial_counter);
      g_serial_counter = 0;
    } else {
      // "buffer_copied" is sent to request new frame from the PC.
      Serial.println("buffer_copied");
      // Make sure that the "buffer_copied" is not send twice in loop().
      return;
    }
  }

  // The "buffer_copied" here will trigger the first frame to be transmitted,
  // when PWM interrupt asks for it.
  // This will kickstart the USB-serial exchange with the PC in
  // Serial.available()
  if (g_sequence_finished) {
    Serial.println("buffer_copied");
    g_sequence_finished = false;
  }
}

void OnPwmSequenceEnd() {
  g_which_pwm_module_triggered = SleeveTactors.GetEvent();

  if (g_which_pwm_module_triggered == 0) {
    int mem_offset =
        kHeaderBytes +
        (g_frame_counter * (kNumChannels * kPwmSamples + kHeaderBytes));
    memcpy(g_received_tactile_frame, g_playback_pwm_buffer + mem_offset,
           (kNumChannels * kPwmSamples));

    SleeveTactors.UpdatePwmAllChannelsByte(g_received_tactile_frame);

    // Trigger the request for new data after playing kWhenToSendFrame frames.
    if (g_frame_counter == kWhenToSendFrame) {
      g_sequence_finished = true;
    }
    ++g_frame_counter;

    // When the last frame is played swap the PWM buffer with new serial data.
    if (g_frame_counter >= kNumBufferedFrames) {
      g_frame_counter = 0;
      memcpy(g_playback_pwm_buffer, g_pending_playback_pwm_buffer,
             (kDataBytes + kHeaderBytes) * kNumBufferedFrames);
    }
  }
}

void FlashLeds() {
  for (int i = 0; i < 5; ++i) {
    nrf_gpio_pin_write(kLedPinBlue, 1);
    nrf_gpio_pin_write(kLedPinGreen, 1);
    delay(100);
    nrf_gpio_pin_write(kLedPinBlue, 0);
    nrf_gpio_pin_write(kLedPinGreen, 0);
    delay(100);
  }
}
