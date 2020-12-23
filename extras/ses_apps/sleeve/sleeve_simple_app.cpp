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
// App for running the tactors on the sleeve.
//
// This app was compiled using Segger studio (SES). The code takes the input
// from the puck through the serial port. Upon startup, the sleeve will start
// listening to the serial port analog mic data. The analog mic data is
// processed by sleeve's tactile processor and outputs pwm to the tactors. The
// audio data is downsampled by 8x for the tactile processor, and after
// processing it is upsampled back (8x) by the pwm module.
//
// The sleeve implements a state machine, controlled by the puck commands.
// Either raw data from the analog microphone or tactor pwm values could be
// sent. Also, the puck can send a command to disable/enable the audio
// amplifiers.
//
// On the puck:
// Pressing button 1 on the puck silences the amplifiers. Pressing button 2 will
// change the mode to test mode, where tactors are turned on one by one.
// Pressing 2 again goes back to analog mic mode.
//
// The timing is as follows:
// Analog mic data is sent every 4 ms, corresponding to 64 samples at 15kHz.
// Data transmission over serial takes 1.3 ms at 1Mbit/sec; this happens while
// mic data is collected. With optimization level 0 it takes 8 ms to do the
// processing (mostly tactor_processor) on the sleeve. With optimization level 2
// for speed (in SES), time is shortened to 3.2 ms.
//
// On occasion, the startup sequence fails, so the sleeve will automatically
// restart. This could take a couple of restart cycles.
//
// The tactors on the flex pcb is ordered as follows below. The op codes allows
// control of the specific channel from the puck. Op code: (PWM module, PWM
// channel). The graphic below has op codes in (x).
//
//   1: (0, 0)    6: (1, 1)                    (4)-(1)
//   2: (0, 1)    7: (1, 2)                    /     \      (2)
//   3: (0, 2)    8: (1, 3)            (3)   (8) (6) (5)
//   4: (0, 3)    9: (2, 0)                    \     /      (9)
//   5: (1, 0)    10:(2, 1)                    (7)-(10)
//
// The tactile processor output is ordered as following:
//
//   0: baseband  5: eh                        (6)-(5)   sh fricative
//   1: aa        6: ae                        /     \      (8)
//   2: uw        7: uh               (0)    (1) (7) (4)
//   3: ih        8: sh fricative   baseband   \     /      (9)
//   4: iy        9: fricative                 (2)-(3)   fricative
//                                          vowel cluster

#include <math.h>
#include <stdio.h>
#include <string.h>

// Hardware drivers.
#include "Lp5012.h"
#include "board_defs.h"              // NOLINT(build/include)
#include "pwm_sleeve.h"              // NOLINT(build/include)
#include "serial_puck_sleeve.h"      // NOLINT(build/include)
#include "temperature_monitor.h"     // NOLINT(build/include)
#include "two_wire.h"                // NOLINT(build/include)
#include "classify_phenome_param.h"  // NOLINT(build/include)
#include "tactile_processor_cpp.h"   // NOLINT(build/include)
#include "post_processor_cpp.h"      // NOLINT(build/include)
#include "look_up.h"                 // NOLINT(build/include)

// FreeRTOS libraries.
#include "FreeRTOS.h"
#include "timers.h"  // NOLINT(build/include)

// nRF libraries.
#include "app_error.h"                 // NOLINT(build/include)
#include "nrf.h"                       // NOLINT(build/include)
#include "nrf_delay.h"                 // NOLINT(build/include)
#include "nrf_log.h"                   // NOLINT(build/include)
#include "nrf_log_ctrl.h"              // NOLINT(build/include)
#include "nrf_log_default_backends.h"  // NOLINT(build/include)

using namespace audio_tactile;  // NOLINT(build/namespaces)

const int kMicSamples = 64;
static_assert(kMicSamples <= (kTxDataSize / 2),
              "Mic data is larger than Tx buffer size");

const int kPwmSamples = 8;
static_assert(kPwmSamples <= kNumPwmValues,
              "Number of pwm samples is larger than pwm buffer size");

const int kPwmSamplesAllChannels = kPwmSamples * kNumTotalPwm;

static TaskHandle_t m_tactor_task_handle;
static TaskHandle_t audio_processor_task_handle;

static int16_t received_mic_data[kMicSamples];

// Buffer of input audio as floats in [-1, 1].
static float audio_input[kMicSamples];

static uint16_t pwm_rx[kPwmSamples];

static uint16_t pwm_all_rx[kPwmSamples * kNumPwmChannels];

static uint8_t which_pwm_module_triggered;

static bool continous_play_back = true;

static bool tactor_processor_on = false;

static bool led_initialized = false;

static int gain;
static int denoising;
static int compression;

TactileProcessorWrapper AudioClassifier;
PostProcessorWrapper AudioPostProcessor;

static float* src;

uint8_t all_tactor_streaming_buffer[kPwmSamplesAllChannels];

void on_PWM_sequence_end() {
  which_pwm_module_triggered = SleeveTactors.GetEvent();

  if (continous_play_back) {
    if (which_pwm_module_triggered == 0) {
      SleeveTactors.UpdatePwmAllChannelsByte(all_tactor_streaming_buffer);

      uint8_t tx_byte[1];
      tx_byte[0] = kRequestMoreData;
      PuckSleeveSerialPort.SendDataRaw(tx_byte, 1);
    }
  }

  if (tactor_processor_on && which_pwm_module_triggered == 0) {
    float pwm_channel[kPwmSamples];

    // Send samples to the tactors. I don't use loops now, since we might
    // adjust the tactor order.
    for (int i = 0; i < kTactileFramesPerCarlBlock; ++i) {
      pwm_channel[i] = src[i * kTactileProcessorNumTactors];
    }
    SleeveTactors.UpdatePwmModuleChannelFloat(pwm_channel, 0, 2);

    for (int i = 0; i < kTactileFramesPerCarlBlock; ++i) {
      pwm_channel[i] = src[(i * kTactileProcessorNumTactors) + 1];
    }
    SleeveTactors.UpdatePwmModuleChannelFloat(pwm_channel, 1, 3);

    for (int i = 0; i < kTactileFramesPerCarlBlock; ++i) {
      pwm_channel[i] = src[(i * kTactileProcessorNumTactors) + 2];
    }
    SleeveTactors.UpdatePwmModuleChannelFloat(pwm_channel, 1, 2);

    for (int i = 0; i < kTactileFramesPerCarlBlock; ++i) {
      pwm_channel[i] = src[(i * kTactileProcessorNumTactors) + 3];
    }
    SleeveTactors.UpdatePwmModuleChannelFloat(pwm_channel, 2, 1);

    for (int i = 0; i < kTactileFramesPerCarlBlock; ++i) {
      pwm_channel[i] = src[(i * kTactileProcessorNumTactors) + 4];
    }
    SleeveTactors.UpdatePwmModuleChannelFloat(pwm_channel, 1, 0);

    for (int i = 0; i < kTactileFramesPerCarlBlock; ++i) {
      pwm_channel[i] = src[(i * kTactileProcessorNumTactors) + 5];
    }
    SleeveTactors.UpdatePwmModuleChannelFloat(pwm_channel, 0, 0);

    for (int i = 0; i < kTactileFramesPerCarlBlock; ++i) {
      pwm_channel[i] = src[(i * kTactileProcessorNumTactors) + 6];
    }
    SleeveTactors.UpdatePwmModuleChannelFloat(pwm_channel, 0, 3);

    for (int i = 0; i < kTactileFramesPerCarlBlock; ++i) {
      pwm_channel[i] = src[(i * kTactileProcessorNumTactors) + 7];
    }
    SleeveTactors.UpdatePwmModuleChannelFloat(pwm_channel, 1, 1);

    for (int i = 0; i < kTactileFramesPerCarlBlock; ++i) {
      pwm_channel[i] = src[(i * kTactileProcessorNumTactors) + 8];
    }
    SleeveTactors.UpdatePwmModuleChannelFloat(pwm_channel, 0, 1);

    for (int i = 0; i < kTactileFramesPerCarlBlock; ++i) {
      pwm_channel[i] = src[(i * kTactileProcessorNumTactors) + 9];
    }
    SleeveTactors.UpdatePwmModuleChannelFloat(pwm_channel, 2, 0);
  }
}

void on_new_serial_data() {
  switch (PuckSleeveSerialPort.GetEvent()) {
    case kLoadMicDataOpCode:
      tactor_processor_on = true;
      continous_play_back = false;
      PuckSleeveSerialPort.GetMicrophoneData(received_mic_data, kMicSamples);
      // Unblock the audio processing task.
      BaseType_t xHigherPriorityTaskWoken;
      xHigherPriorityTaskWoken = pdFALSE;
      vTaskNotifyGiveFromISR(audio_processor_task_handle,
                             &xHigherPriorityTaskWoken);
      portEND_SWITCHING_ISR(xHigherPriorityTaskWoken);
      break;
    case kLoadTactorL1:
      tactor_processor_on = false;
      PuckSleeveSerialPort.GetPlayOneTactorData(pwm_rx, kPwmSamples);
      SleeveTactors.UpdatePwmModuleChannel(pwm_rx, 0, 0);
      break;
    case kLoadTactorR1:
      tactor_processor_on = false;
      PuckSleeveSerialPort.GetPlayOneTactorData(pwm_rx, kPwmSamples);
      SleeveTactors.UpdatePwmModuleChannel(pwm_rx, 0, 1);
      // this tactor for teseting.
      break;
    case kLoadTactorL2:
      tactor_processor_on = false;
      PuckSleeveSerialPort.GetPlayOneTactorData(pwm_rx, kPwmSamples);
      SleeveTactors.UpdatePwmModuleChannel(pwm_rx, 0, 2);
      break;
    case kLoadTactorR2:
      tactor_processor_on = false;
      PuckSleeveSerialPort.GetPlayOneTactorData(pwm_rx, kPwmSamples);
      SleeveTactors.UpdatePwmModuleChannel(pwm_rx, 0, 3);
      break;
    case kLoadTactorL3:
      tactor_processor_on = false;
      PuckSleeveSerialPort.GetPlayOneTactorData(pwm_rx, kPwmSamples);
      SleeveTactors.UpdatePwmModuleChannel(pwm_rx, 1, 0);
      break;
    case kLoadTactorR3:
      tactor_processor_on = false;
      PuckSleeveSerialPort.GetPlayOneTactorData(pwm_rx, kPwmSamples);
      SleeveTactors.UpdatePwmModuleChannel(pwm_rx, 1, 1);
      break;
    case kLoadTactorL4:
      tactor_processor_on = false;
      PuckSleeveSerialPort.GetPlayOneTactorData(pwm_rx, kPwmSamples);
      SleeveTactors.UpdatePwmModuleChannel(pwm_rx, 1, 2);
      break;
    case kLoadTactorR4:
      tactor_processor_on = false;
      PuckSleeveSerialPort.GetPlayOneTactorData(pwm_rx, kPwmSamples);
      SleeveTactors.UpdatePwmModuleChannel(pwm_rx, 1, 3);
      break;
    case kLoadTactorL5:
      tactor_processor_on = false;
      PuckSleeveSerialPort.GetPlayOneTactorData(pwm_rx, kPwmSamples);
      SleeveTactors.UpdatePwmModuleChannel(pwm_rx, 2, 0);
      break;
    case kLoadTactorR5:
      tactor_processor_on = false;
      PuckSleeveSerialPort.GetPlayOneTactorData(pwm_rx, kPwmSamples);
      SleeveTactors.UpdatePwmModuleChannel(pwm_rx, 2, 1);
      break;
    case kTurnOffAmplifiers:
      SleeveTactors.DisableAmplifiers();
      break;
    case kTurnOnAmplifiers:
      SleeveTactors.EnableAmplifiers();
      break;
    case kSetOutputGain:
      gain = PuckSleeveSerialPort.GetTuningParameters();
      AudioClassifier.SetOutputGain(gain);
      if (!led_initialized) {
        LedArray.Initialize();
        led_initialized = true;
      }
      LedArray.Clear();
      LedArray.LedBar(gain, 20);
      break;
    case kSetDenoising:
      denoising = PuckSleeveSerialPort.GetTuningParameters();
      AudioClassifier.SetDenoising(denoising);
      if (!led_initialized) {
        LedArray.Initialize();
        led_initialized = true;
      }
      LedArray.Clear();
      LedArray.SetOneLed(10, 100);
      LedArray.LedBar(denoising, 20);
      break;
    case kSetCompression:
      compression = PuckSleeveSerialPort.GetTuningParameters();
      AudioClassifier.SetCompression(compression);
      if (!led_initialized) {
        LedArray.Initialize();
        led_initialized = true;
      }
      LedArray.Clear();
      LedArray.SetOneLed(11, 100);
      LedArray.LedBar(compression, 20);
      break;
    case kStreamDataStart:
      SleeveTactors.DisableAmplifiers();
      continous_play_back = true;
      break;
    case kStreamDataStop:
      SleeveTactors.EnableAmplifiers();
      continous_play_back = false;
      break;
    case kPlayAllChannels:
      PuckSleeveSerialPort.GetPlayAllTactorsData(all_tactor_streaming_buffer,
                                                 kPwmSamplesAllChannels);
      continous_play_back = true;
    case kCommError:
      // Maybe reset if there are errors. Serial coms are not always reliable.
      break;
    case kTimeOutError:
      NVIC_SystemReset();
      break;
    default:
      // Handle an unknown op code event.
      NRF_LOG_RAW_INFO("== UNKNOWN OPCODE ==\n");
      NRF_LOG_FLUSH();
      break;
  }
}

// This interrupt is triggered on overheating.
void on_overheating() { nrf_gpio_pin_toggle(kLedPin); }

static void audio_processor_task_function(void* pvParameter) {
  for (;;) {
    // Block the task until interrupt is triggered, when new data is collected.
    // https://www.freertos.org/xTaskNotifyGive.html
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    nrf_gpio_pin_write(kLedPin, 1);

    // Convert ADC values to floats in range of [-1, 1].
    // The raw values are centered around 1500, so to zero them we need to
    // subtract 1500. The raw ADC values can swing from -2048 to 2048, so we
    // scale to that value.
    for (int i = 0; i < kMicSamples; ++i) {
      audio_input[i] = ((float)(received_mic_data[i]) - 1500.0f) / 2048.0f;
    }

    // Process samples.
    src = AudioClassifier.ProcessSamples(audio_input);
    AudioPostProcessor.PostProcessSamples(src);

    nrf_gpio_pin_write(kLedPin, 0);
  }
}

int main() {
  // Initialize log, allows to print to the console with seggers j-link.
  log_init();
  NRF_LOG_RAW_INFO("== SLEEVE START ==\n");

  // Print important constants.
  const float update_period_s =
      (float)kSaadcFramesPerPeriod / kSaadcSampleRateHz;
  NRF_LOG_RAW_INFO(
      "%d us period \n %d mic samples \n %d tactile samples \n %d PWM samples "
      "\n",
      (int)(1e6f * update_period_s + 0.5f), kMicSamples,
      kPwmSamples * kTactileDecimationFactor, kPwmSamples);

  NRF_LOG_FLUSH();

  // Set the indicator led pin to output.
  nrf_gpio_cfg_output(kLedPin);

  // Initialize LED driver.
  // For unknown reason LED driver need to be initialized again in the loop.
  LedArray.Initialize();
  LedArray.CycleAllLeds(1);
  LedArray.SetOneLed(1, 10);

  // Initialize tactor driver.
  SleeveTactors.Initialize();
  SleeveTactors.SetUpsamplingFactor(kTactileDecimationFactor);
  SleeveTactors.OnSequenceEnd(on_PWM_sequence_end);
  SleeveTactors.StartPlayback();

  // Initialize temperature monitor.
  SleeveTemperatureMonitor.StartMonitoringTemperature();
  SleeveTemperatureMonitor.OnOverheatingEventListener(on_overheating);

  // Initialize the tactile processor.
  AudioClassifier.Init(kSaadcSampleRateHz, kCarlBlockSize,
                       kTactileDecimationFactor);

  const float kDefaultGain = 4.0f;
  AudioPostProcessor.Init(AudioClassifier.GetOutputSampleRate(),
                          AudioClassifier.GetOutputBlockSize(),
                          AudioClassifier.GetOutputNumberTactileChannels(),
                          kDefaultGain);

  NRF_LOG_RAW_INFO("== TACTILE PROCESSOR SETUP DONE ==\n");
  NRF_LOG_FLUSH();

  // Initialize serial port.
  PuckSleeveSerialPort.InitializeSleeve();
  PuckSleeveSerialPort.OnSerialDataReceived(on_new_serial_data);

  // Start freeRTOS.
  xTaskCreate(audio_processor_task_function, "audio",
              configMINIMAL_STACK_SIZE + 500, NULL, 2,
              &audio_processor_task_handle);
  vTaskStartScheduler();
  // App should go into tasks naw, and not reach the code below. Reset the
  // system if it does.
  NVIC_SystemReset();
}

void HardFault_Handler(void) {
  uint32_t* sp = (uint32_t*)__get_MSP();  // Get stack pointer.
  uint32_t ia = sp[12];                   // Get instruction address from stack.

  NRF_LOG_RAW_INFO("Hard Fault at address: 0x%08x\r\n", (unsigned int)ia);
  NRF_LOG_FLUSH();
  while (true) {
  }
}

static void log_init(void) {
  ret_code_t err_code = NRF_LOG_INIT(NULL);
  APP_ERROR_CHECK(err_code);
  NRF_LOG_DEFAULT_BACKENDS_INIT();
}
