// Copyright 2020-2021 Google LLC
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

// NOLINTBEGIN(build/include)

#include <math.h>
#include <stdio.h>
#include <string.h>

// Hardware drivers.
#include "analog_external_mic.h"
#include "board_defs.h"
#include "lp5012.h"
#include "pwm_sleeve.h"
#include "serial_com.h"

// TODO: There is build error for temperature_monitor and
// battery_monitor, as they use the same LPCOMP_COMP_IRQHandler.
// Should find a way to resolve it.

#include "look_up.h"
#include "post_processor_cpp.h"
#include "dsp/channel_map.h"
#include "tactile/tactile_pattern.h"
#include "tactile_processor_cpp.h"
#include "two_wire.h"

// FreeRTOS libraries.
#include "FreeRTOS.h"
#include "timers.h"

// nRF libraries.
#include "app_error.h"
#include "nrf.h"
#include "nrf_delay.h"
#include "nrf_log.h"
#include "nrf_log_ctrl.h"
#include "nrf_log_default_backends.h"
#include "nrf_uarte.h"

// NOLINTEND

using namespace audio_tactile;  // NOLINT(build/namespaces)

const int kPwmSamplesAllChannels = kNumPwmValues * kNumTotalPwm;

// Audio samples per CARL block. Must be no more than kSaadcFramesPerPeriod.
const int kCarlBlockSize = 64;
// Decimation factor in audio-to-tactile processing. Tactile signals are
// bandlimited below 500 Hz, so e.g. 8x decimation is reasonable.
const int kTactileDecimationFactor = 8;
// Number of tactile frames per CARL block.
const int kTactileFramesPerCarlBlock =
    kCarlBlockSize / kTactileDecimationFactor;

static TaskHandle_t g_tactile_processor_task_handle;

// Buffer of received mic audio as int16 samples.
static int16_t g_mic_audio_int16[kAdcDataSize];
// Buffer of mic audio converted to floats in [-1, 1].
static float g_mic_audio_float[kAdcDataSize];

static uint16_t pwm_rx[kNumPwmValues];

// True when serial is streaming tactile playback.
static bool g_streaming_tactile_playback = true;
// True when serial is receiving mic audio (kLoadMicDataOpCode).
static bool g_receiving_audio = false;
static bool g_led_initialized = false;

// TactileProcessor, turns audio into tactile signals.
TactileProcessorWrapper g_tactile_processor;
// Tuning knobs for configuring tactile processing.
TuningKnobs g_tuning_knobs;
// TactilePostprocessor, does equalization and limiting on tactile signals.
PostProcessorWrapper g_post_processor;
// Channel to tactor mapping and final output gains.
ChannelMap g_channel_map;

// Tactile pattern synthesizer.
TactilePattern g_tactile_pattern;

// Pointer to the tactile output buffer of g_tactile_processor.
static float* g_tactile_output;

// Buffer for streaming tactile playback.
uint8_t g_all_tactor_streaming_buffer[kPwmSamplesAllChannels];

static void OnSerialEvent();
static void LogInit();
void OnPwmSequenceEnd();
void OnOverheating();
static void TactileProcessorTaskFun(void* pvParameter);

void OnPwmSequenceEnd() {
  const int which_pwm_module_triggered = SleeveTactors.GetEvent();
  // All modules are used together, so only take action on the module 0 trigger.
  if (which_pwm_module_triggered != 0) { return; }

  if (g_streaming_tactile_playback) {
      SleeveTactors.UpdatePwmAllChannelsByte(g_all_tactor_streaming_buffer);
      // Transmit one byte to synchronize.
      uint8_t tx_byte[1];
      tx_byte[0] = static_cast<uint8_t>(MessageType::kRequestMoreData);
      SerialCom.SendRaw(Slice<uint8_t, 1>(tx_byte));
  } else {
    constexpr int kNumChannels = 10;
    // Hardware channel `c` plays logical channel kHwToLogical[c].
    constexpr static int kHwToLogical[kNumChannels] = {5, 8, 0, 6, 4,
                                                       7, 2, 1, 9, 3};

    // There are two levels of mapping:
    // 1. Hardware channel `c` plays logical channel kHwToLogical[c],
    //      hw[c] = logical[kHwToLogical[c]].
    // 2. g_channel_map maps tactile channels to logical channels,
    //      logical[c] = channel_map.gains[c] * tactile[channel_map.sources[c]].
    //
    // We compose this into a single step of copying as
    //   hw[c] = channel_map.gains[logical_c]
    //           * tactile[channel_map.sources[logical_c]].
    // with logical_c = kHwToLogical[c].
    for (int c = 0; c < kNumChannels; ++c) {
      const int logical_c = kHwToLogical[c];
      const float gain = g_channel_map.gains[logical_c];
      const float* src = g_tactile_output + g_channel_map.sources[logical_c];
      SleeveTactors.UpdateChannelWithGain(c, gain, src,
                                          kTactileProcessorNumTactors);
    }
  }
}

void HandleMessage(const Message& message) {
  switch (message.type()) {
    case MessageType::kAudioSamples: {
      g_receiving_audio = true;
      g_streaming_tactile_playback = false;
      message.ReadAudioSamples(
          Slice<int16_t, kAdcDataSize>(g_mic_audio_int16));
      // Unblock the audio processing task.
      BaseType_t xHigherPriorityTaskWoken;
      xHigherPriorityTaskWoken = pdFALSE;
      vTaskNotifyGiveFromISR(g_tactile_processor_task_handle,
                             &xHigherPriorityTaskWoken);
      portEND_SWITCHING_ISR(xHigherPriorityTaskWoken);
    } break;
    case MessageType::kTactor1Samples:
    case MessageType::kTactor2Samples:
    case MessageType::kTactor3Samples:
    case MessageType::kTactor4Samples:
    case MessageType::kTactor5Samples:
    case MessageType::kTactor6Samples:
    case MessageType::kTactor7Samples:
    case MessageType::kTactor8Samples:
    case MessageType::kTactor9Samples:
    case MessageType::kTactor10Samples:
    case MessageType::kTactor11Samples:
    case MessageType::kTactor12Samples: {
      int channel;
      if (message.ReadSingleTactorSamples(
              &channel, Slice<uint16_t, kNumPwmValues>(pwm_rx))) {
        SleeveTactors.UpdateChannel(channel - 1, pwm_rx);
      }
    } break;
    case MessageType::kDisableAmplifiers:
      SleeveTactors.DisableAmplifiers();
      break;
    case MessageType::kEnableAmplifiers:
      SleeveTactors.EnableAmplifiers();
      break;
    case MessageType::kTuning: {
      if (message.ReadTuning(&g_tuning_knobs)) {
        g_tactile_processor.ApplyTuning(g_tuning_knobs);
        if (!g_led_initialized) {
          LedArray.Initialize();
          g_led_initialized = true;
        }
        LedArray.Clear();
        LedArray.LedBar(g_tuning_knobs.values[kKnobOutputGain], 20);

        // Play "confirm" pattern as UI feedback when new settings are applied.
        TactilePatternStart(&g_tactile_pattern, kTactilePatternConfirm);
      }
    } break;
    case MessageType::kTactilePattern: {
      char simple_pattern[kMaxTactilePatternLength + 1];
      if (message.ReadTactilePattern(simple_pattern)) {
        TactilePatternStart(&g_tactile_pattern, simple_pattern);
      }
    } break;
    case MessageType::kChannelMap:
      message.ReadChannelMap(&g_channel_map,
                             kTactileProcessorNumTactors,
                             kTactileProcessorNumTactors);
      break;
    case MessageType::kChannelGainUpdate: {
      int test_channels[2];
      if (message.ReadChannelGainUpdate(&g_channel_map, test_channels,
                                        kTactileProcessorNumTactors,
                                        kTactileProcessorNumTactors)) {
        TactilePatternStartCalibrationTones(&g_tactile_pattern,
                                            test_channels[0], test_channels[1]);
      }
    } break;
    case MessageType::kStreamDataStart:
      SleeveTactors.DisableAmplifiers();  // Should this be Enable?
      g_streaming_tactile_playback = true;
      break;
    case MessageType::kStreamDataStop:
      SleeveTactors.EnableAmplifiers();  // Should this be Disable?
      g_streaming_tactile_playback = false;
      break;
    case MessageType::kAllTactorsSamples:
      if (message.ReadAllTactorsSamples(
              Slice<uint8_t, kNumTotalPwm * kNumPwmValues>(
                  g_all_tactor_streaming_buffer))) {
        g_streaming_tactile_playback = true;
      }
      break;
    default:
      // Handle an unknown op code event.
      NRF_LOG_RAW_INFO("== UNKNOWN MESSAGE TYPE ==\n");
      NRF_LOG_FLUSH();
      break;
  }
}

static void OnSerialEvent() {
  g_receiving_audio = false;
  switch (SerialCom.event()) {
    case SerialEvent::kMessageReceived:
      HandleMessage(SerialCom.rx_message());
      break;
    case SerialEvent::kCommError:
      // Maybe reset if there are errors. Serial coms are not always reliable.
      break;
    case SerialEvent::kTimeOutError:
      // Disable task auto-restarting.
      nrf_uarte_shorts_disable(NRF_UARTE0, NRF_UARTE_SHORT_ENDRX_STARTRX);
      // Stop UARTE Rx task.
      nrf_uarte_task_trigger(NRF_UARTE0, NRF_UARTE_TASK_STOPRX);
      nrf_delay_ms(8);  // Wait a couple frames.
      // Re-enable auto-restarting.
      nrf_uarte_shorts_enable(NRF_UARTE0, NRF_UARTE_SHORT_ENDRX_STARTRX);
      // Start UARTE Rx task.
      nrf_uarte_task_trigger(NRF_UARTE0, NRF_UARTE_TASK_STARTRX);
      break;
    default:
      NRF_LOG_RAW_INFO("== UNKNOWN SERIAL EVENT ==\n");
      NRF_LOG_FLUSH();
      break;
  }
}

// This interrupt is triggered on overheating.
void OnOverheating() { nrf_gpio_pin_toggle(kLedPinBlue); }

static void TactileProcessorTaskFun(void*) {
  for (;;) {
    // Block the task until interrupt is triggered, when new data is collected.
    // https://www.freertos.org/xTaskNotifyGive.html
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    nrf_gpio_pin_write(kLedPinBlue, 1);

    // Play synthesized tactile pattern if either a pattern is active or if the
    // sleeve is not receiving audio, for instance because of a timeout error.
    // If the pattern isn't active, the pattern synthesizer produces silence.
    if (!g_receiving_audio || TactilePatternIsActive(&g_tactile_pattern)) {
      TactilePatternSynthesize(
          &g_tactile_pattern, g_tactile_processor.GetOutputBlockSize(),
          g_tactile_output);
    } else {
      // Convert ADC values to floats. The raw ADC values can swing from -2048
      // to 2048, so we scale by that value.
      const float scale = TuningGetInputGain(&g_tuning_knobs) / 2048.0f;
      for (int i = 0; i < kAdcDataSize; ++i) {
        g_mic_audio_float[i] = scale * g_mic_audio_int16[i];
      }
      // Process samples.
      g_tactile_output = g_tactile_processor.ProcessSamples(g_mic_audio_float);
    }

    g_post_processor.PostProcessSamples(g_tactile_output);

    nrf_gpio_pin_write(kLedPinBlue, 0);
  }
}

static void LogInit(void) {
  ret_code_t err_code = NRF_LOG_INIT(nullptr);
  APP_ERROR_CHECK(err_code);
  NRF_LOG_DEFAULT_BACKENDS_INIT();
}

int main() {
  // Initialize log, allows to print to the console with seggers j-link.
  LogInit();
  NRF_LOG_RAW_INFO("== SLEEVE START ==\n");

  const int kPwmFramesPerPeriod = 64;
  // SAADC sample rate in Hz.
  const int kSaadcSampleRateHz = 15625;
  // To simplify buffering logic, we determine kSaadcFramesPerPeriod so that the
  // SAADC update period is equal to the PWM update period.
  const float kSaadcFramesPerPeriod =
      ((2 * kSaadcSampleRateHz * 512 * kPwmFramesPerPeriod) / 16000000L);

  // Print important constants.
  const float update_period_s =
      (float)kSaadcFramesPerPeriod / kSaadcSampleRateHz;
  NRF_LOG_RAW_INFO(
      "%d us period \n %d mic samples \n %d tactile samples \n %d PWM samples "
      "\n",
      (int)(1e6f * update_period_s + 0.5f), kAdcDataSize,
      kNumPwmValues * kTactileDecimationFactor, kNumPwmValues);

  NRF_LOG_FLUSH();

  // Set the indicator led pin to output.
  nrf_gpio_cfg_output(kLedPinBlue);

  // Initialize LED driver.
  // For unknown reason LED driver need to be initialized again in the loop.
  LedArray.Initialize();
  LedArray.CycleAllLeds(1);
  LedArray.SetOneLed(1, 10);

  // Initialize tactor driver.
  SleeveTactors.OnSequenceEnd(OnPwmSequenceEnd);
  SleeveTactors.Initialize();
  SleeveTactors.SetUpsamplingFactor(kTactileDecimationFactor);
  SleeveTactors.StartPlayback();

  // Initialize temperature monitor.
  // SleeveTemperatureMonitor.StartMonitoringTemperature();
  // SleeveTemperatureMonitor.OnOverheatingEventListener(OnOverheating);

  // Initialize the tactile processor.
  g_tactile_processor.Init(kSaadcSampleRateHz, kCarlBlockSize,
                           kTactileDecimationFactor);

  const float kDefaultGain = 4.0f;
  g_post_processor.Init(g_tactile_processor.GetOutputSampleRate(),
                        g_tactile_processor.GetOutputBlockSize(),
                        g_tactile_processor.GetOutputNumberTactileChannels(),
                        kDefaultGain);
  ChannelMapInit(&g_channel_map, kTactileProcessorNumTactors);

  TactilePatternInit(&g_tactile_pattern,
                     g_tactile_processor.GetOutputSampleRate(),
                     g_tactile_processor.GetOutputNumberTactileChannels());

  NRF_LOG_RAW_INFO("== TACTILE PROCESSOR SETUP DONE ==\n");
  NRF_LOG_FLUSH();

  // Initialize serial port.
  SerialCom.InitSleeve(OnSerialEvent);

  // Start freeRTOS.
  xTaskCreate(TactileProcessorTaskFun, "audio", configMINIMAL_STACK_SIZE + 500,
              nullptr, 2, &g_tactile_processor_task_handle);
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
