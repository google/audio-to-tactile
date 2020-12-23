/* Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tactile_api_handlers.h" /* NOLINT(build/include) */

#include <pb_decode.h>
#include <pb_encode.h>

#include "app_config.h"  /* NOLINT(build/include) */
#include "nrf_log.h"     /* NOLINT(build/include) */
#include "nus_message.h" /* NOLINT(build/include) */

Tactor tactor_info[kTactileProcessorNumTactors] = {
    (Tactor){.has_x = true,
             .x = 30.0,
             .has_y = true,
             .y = 10.0,
             .has_note = true,
             .note = "baseband"},
    (Tactor){.has_x = true,
             .x = 30.0,
             .has_y = true,
             .y = 30.0,
             .has_note = true,
             .note = "aa"},
    (Tactor){.has_x = true,
             .x = 10.0,
             .has_y = true,
             .y = 40.0,
             .has_note = true,
             .note = "uw"},
    (Tactor){.has_x = true,
             .x = 10.0,
             .has_y = true,
             .y = 60.0,
             .has_note = true,
             .note = "ih"},
    (Tactor){.has_x = true,
             .x = 30.0,
             .has_y = true,
             .y = 70.0,
             .has_note = true,
             .note = "iy"},
    (Tactor){.has_x = true,
             .x = 50.0,
             .has_y = true,
             .y = 60.0,
             .has_note = true,
             .note = "eh"},
    (Tactor){.has_x = true,
             .x = 50.0,
             .has_y = true,
             .y = 40.0,
             .has_note = true,
             .note = "ae"},
    (Tactor){.has_x = true,
             .x = 30.0,
             .has_y = true,
             .y = 50.0,
             .has_note = true,
             .note = "uh"},
    (Tactor){.has_x = true,
             .x = 40.0,
             .has_y = true,
             .y = 90.0,
             .has_note = true,
             .note = "fricative"},
    (Tactor){.has_x = true,
             .x = 20.0,
             .has_y = true,
             .y = 90.0,
             .has_note = true,
             .note = "sh fricative"}};

/* Default operation mode */
OperationMode current_operation_mode = {
    .has_input = true,
    .input = OperationMode_InputOption_ANALOG_EXTERNAL_MIC,
    .has_process = true,
    .process = OperationMode_ProcessOption_AUDIO_TO_TACTILE,
    .has_output = true,
    .output = OperationMode_OutputOption_SERIAL};

/* List of all supported operation modes */
OperationMode supported_operation_modes[] = {
    (OperationMode){.has_input = true,
                    .input = OperationMode_InputOption_ANALOG_EXTERNAL_MIC,
                    .has_process = true,
                    .process = OperationMode_ProcessOption_AUDIO_TO_TACTILE,
                    .has_output = true,
                    .output = OperationMode_OutputOption_SERIAL},
    {.has_input = true,
     .input = OperationMode_InputOption_BLE_AUDIO,
     .has_process = true,
     .process = OperationMode_ProcessOption_AUDIO_TO_TACTILE,
     .has_output = true,
     .output = OperationMode_OutputOption_SERIAL}};

int output_gain = 100;
int denoising = 200;
int compression = 300;

float tactor_amplitudes[kTactileProcessorNumTactors] = {
    0.7, 0.8, 0.4, 0.5, 0.6, 0.9, 1.0, 0.1, 0.2, 0.3};

float tactor_phases[kTactileProcessorNumTactors] = {0.5, 0.6, 0.9, 0.3, 0.7,
                                                    0.8, 0.4, 1.0, 0.1, 0.2};

float tactor_frequencies[kTactileProcessorNumTactors] = {
    100, 50, 120, 40, 80, 150, 110, 130, 240, 280};

void GetTactorHardwareLayoutResponse() {
  NRF_LOG_DEBUG("GetTactorHardwareLayoutResponse %ld");
  nus_send_message = (TactileMessage){
      .which_type = TactileMessage_get_response_tag,
      .type.get_response = (GetResponse){
          .which_type = GetResponse_tactor_hardware_layout_response_tag,
          .type.tactor_hardware_layout_response =
              TactorHardwareLayout_init_zero}};
  nus_send_message.type.get_response.type.tactor_hardware_layout_response
      .tactors_count = kTactileProcessorNumTactors;
  /* Not exactly sure how union and also an array type works but it seems memcpy
   * does not work well in setting values in union.
   */
  for (int i = 0; i < kTactileProcessorNumTactors; ++i) {
    nus_send_message.type.get_response.type.tactor_hardware_layout_response
        .tactors[i] = tactor_info[i];
  }

  if (EncodeTactileMessage()) {
    SendTactileMessage();
  } else {
    NRF_LOG_DEBUG("GetTactorHardwareLayoutResponse encode error");
  }
}

void GetBatteryResponse() {
  NRF_LOG_DEBUG("GetBatteryResponse");
  nus_send_message =
      (TactileMessage){.which_type = TactileMessage_get_response_tag,
                     .type.get_response = (GetResponse){
                         .which_type = GetResponse_battery_response_tag,
                         /* placeholder for now, replace with real value */
                         .type.battery_response = 90}};
  if (EncodeTactileMessage()) {
    SendTactileMessage();
  } else {
    NRF_LOG_DEBUG("GetBatteryResponse encode error");
  }
}

void GetSupportedOperationModesResponse() {
  NRF_LOG_DEBUG("GetSupportedOperationModesResponse");
  nus_send_message = (TactileMessage){
      .which_type = TactileMessage_get_response_tag,
      .type.get_response = (GetResponse){
          .which_type = SupportedOperationModes_supported_operation_modes_tag,
          .type.supported_operation_modes_response =
              SupportedOperationModes_init_zero}};
  int num_operation_modes =
      sizeof(supported_operation_modes) / sizeof(supported_operation_modes[0]);
  nus_send_message.type.get_response.type.supported_operation_modes_response
      .supported_operation_modes_count = num_operation_modes;
  for (int i = 0; i < num_operation_modes; ++i) {
    nus_send_message.type.get_response.type.supported_operation_modes_response
        .supported_operation_modes[i] = supported_operation_modes[i];
  }

  if (EncodeTactileMessage()) {
    SendTactileMessage();
  } else {
    NRF_LOG_DEBUG("GetSupportedOperationModesResponse encode error");
  }
}
void GetCurrentOperationModeResponse() {
  NRF_LOG_DEBUG("GetCurrentOperationModeResponse");
  nus_send_message = (TactileMessage){
      .which_type = TactileMessage_get_response_tag,
      .type.get_response = (GetResponse){
          .which_type = GetResponse_current_operation_mode_response_tag,
          .type.current_operation_mode_response = current_operation_mode}};
  if (EncodeTactileMessage()) {
    SendTactileMessage();
  } else {
    NRF_LOG_DEBUG("GetCurrentOperationModeResponse encode error");
  }
}
void GetOperationIsRunningResponse() {
  NRF_LOG_DEBUG("GetOperationIsRunningResponse");
  /* nus_send_message = (TactileMessage) {
   *   .which_type = TactileMessage_get_response_tag,
   *   .type.get_response = (GetResponse) {
   *     .which_type =
   *   }
   * };
   */
  if (EncodeTactileMessage()) {
    SendTactileMessage();
  } else {
    NRF_LOG_DEBUG("GetOperationIsRunningResponse encode error");
  }
}
void GetOutputGainResponse() {
  NRF_LOG_DEBUG("GetOutputGainResponse %d", output_gain);
  nus_send_message =
      (TactileMessage){.which_type = TactileMessage_get_response_tag,
                     .type.get_response = (GetResponse){
                         .which_type = GetResponse_output_gain_response_tag,
                         .type.output_gain_response = output_gain}};
  if (EncodeTactileMessage()) {
    SendTactileMessage();
  } else {
    NRF_LOG_DEBUG("GetOutputGainResponse encode error");
  }
}
void GetDenoisingResponse() {
  NRF_LOG_DEBUG("GetDenoisingResponse %d", denoising);
  nus_send_message =
      (TactileMessage){.which_type = TactileMessage_get_response_tag,
                     .type.get_response = (GetResponse){
                         .which_type = GetResponse_denoising_response_tag,
                         .type.denoising_response = denoising}};
  if (EncodeTactileMessage()) {
    SendTactileMessage();
  } else {
    NRF_LOG_DEBUG("GetDenoisingResponse encode error");
  }
}
void GetCompressionResponse() {
  NRF_LOG_DEBUG("GetCompressionResponse %d", compression);
  nus_send_message =
      (TactileMessage){.which_type = TactileMessage_get_response_tag,
                     .type.get_response = (GetResponse){
                         .which_type = GetResponse_compression_response_tag,
                         .type.compression_response = compression}};
  if (EncodeTactileMessage()) {
    SendTactileMessage();
  } else {
    NRF_LOG_DEBUG("GetCompressionResponse encode error");
  }
}
void GetTactorAmplitudesResponse() {
  NRF_LOG_DEBUG("GetTactorAmplitudesResponse");
  nus_send_message = (TactileMessage){
      .which_type = TactileMessage_get_response_tag,
      .type.get_response = (GetResponse){
          .which_type = GetResponse_tactor_amplitudes_response_tag,
          .type.tactor_amplitudes_response = TactorAmplitudes_init_zero}};
  nus_send_message.type.get_response.type.tactor_amplitudes_response
      .tactor_amplitudes_count = kTactileProcessorNumTactors;
  for (int i = 0; i < kTactileProcessorNumTactors; ++i) {
    nus_send_message.type.get_response.type.tactor_amplitudes_response
        .tactor_amplitudes[i] = tactor_amplitudes[i];
  }
  if (EncodeTactileMessage()) {
    SendTactileMessage();
  } else {
    NRF_LOG_DEBUG("GetTactorAmplitudesResponse encode error");
  }
}
void GetTactorPhasesResponse() {
  NRF_LOG_DEBUG("GetTactorPhasesResponse");
  nus_send_message = (TactileMessage){
      .which_type = TactileMessage_get_response_tag,
      .type.get_response =
          (GetResponse){.which_type = GetResponse_tactor_phases_response_tag,
                        .type.tactor_phases_response = TactorPhases_init_zero}};
  nus_send_message.type.get_response.type.tactor_phases_response
      .tactor_phases_count = kTactileProcessorNumTactors;
  for (int i = 0; i < kTactileProcessorNumTactors; ++i) {
    nus_send_message.type.get_response.type.tactor_phases_response
        .tactor_phases[i] = tactor_phases[i];
  }
  if (EncodeTactileMessage()) {
    SendTactileMessage();
  } else {
    NRF_LOG_DEBUG("GetTactorPhasesResponse encode error");
  }
}
void GetTactorFrequencies() {
  NRF_LOG_DEBUG("GetTactorFrequencies");
  nus_send_message = (TactileMessage){
      .which_type = TactileMessage_get_response_tag,
      .type.get_response = (GetResponse){
          .which_type = GetResponse_tactor_frequencies_response_tag,
          .type.tactor_frequencies_response = TactorFrequencies_init_zero}};
  nus_send_message.type.get_response.type.tactor_frequencies_response
      .tactor_frequencies_count = kTactileProcessorNumTactors;
  for (int i = 0; i < kTactileProcessorNumTactors; ++i) {
    nus_send_message.type.get_response.type.tactor_frequencies_response
        .tactor_frequencies[i] = tactor_frequencies[i];
  }
  if (EncodeTactileMessage()) {
    SendTactileMessage();
  } else {
    NRF_LOG_DEBUG("GetTactorFrequencies encode error");
  }
}

void SetOutputGain(int new_output_gain) {
  NRF_LOG_DEBUG("SetOutputGain: %d", new_output_gain);
  output_gain = new_output_gain;
}

void SetDenoising(int new_denoising) {
  NRF_LOG_DEBUG("SetDenoising: %d", new_denoising);
  denoising = new_denoising;
}

void SetCompression(int new_compression) {
  NRF_LOG_DEBUG("SetCompression: %d", new_compression);
  compression = new_compression;
}

void SetTactorAmplitudes(TactorAmplitudes* tactor_amplitudes_p) {
  NRF_LOG_DEBUG("SetTactorAmplitude");
  int count = tactor_amplitudes_p->tactor_amplitudes_count;
  for (int i = 0; i < count; ++i) {
    tactor_amplitudes[i] = (tactor_amplitudes_p->tactor_amplitudes)[i];
  }
}

void SetTactorPhases(TactorPhases* tactor_phases_p) {
  NRF_LOG_DEBUG("SetTactorPhases");
  int count = tactor_phases_p->tactor_phases_count;
  for (int i = 0; i < count; ++i) {
    tactor_phases[i] = (tactor_phases_p->tactor_phases)[i];
  }
}

void SetTactorFrequencies(TactorFrequencies* tactor_frequencies_p) {
  NRF_LOG_DEBUG("SetTactorPhases");
  int count = tactor_frequencies_p->tactor_frequencies_count;
  for (int i = 0; i < count; ++i) {
    tactor_frequencies[i] = (tactor_frequencies_p->tactor_frequencies)[i];
  }
}
