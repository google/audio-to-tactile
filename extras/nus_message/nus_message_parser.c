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

#include "nus_message_parser.h" /* NOLINT(build/include) */

#include <pb_decode.h>
#include <pb_encode.h>
#include <string.h>

#include "app_config.h"              /* NOLINT(build/include) */
#include "app_scheduler.h"           /* NOLINT(build/include) */
#include "nrf_log.h"                 /* NOLINT(build/include) */
#include "nus_message.h"             /* NOLINT(build/include) */
#include "tactile_api_handlers.h"    /* NOLINT(build/include) */
#include "tactile_ble_service_nus.h" /* NOLINT(build/include) */
#include "tactile_data.pb.h"         /* NOLINT(build/include) */

static void HandleGetRequest(GetRequest_GetOpion option);
static void HandleSetRequest(SetRequest* set_request);

void ParseNusMessage(uint8_t* buffer, uint16_t length) {
  NRF_LOG_INFO("nus length: %u", length);
  /* Parse nanopb message. */
  pb_istream_t stream = pb_istream_from_buffer(buffer, length);
  bool pb_decode_success =
      pb_decode(&stream, ClientMessage_fields, &received_message);
  if (!pb_decode_success) {
    NRF_LOG_INFO("Invalid proto: %s", PB_GET_ERROR(&stream));
    ClearReceivedMessage();
    return;
  }
  switch (received_message.which_type) {
    case ClientMessage_get_request_tag: {
      if (received_message.type.get_request.has_option) {
        HandleGetRequest(received_message.type.get_request.option);
      } else {
        /* Error handling */
      }
    } break;
    case ClientMessage_set_request_tag: {
      SetRequest set_request = received_message.type.set_request;
      HandleSetRequest(&set_request);
    } break;
    case ClientMessage_ble_audio_tag: {
    } break;
    case ClientMessage_ble_fm_generated_waveform_tag: {
    } break;
    default: {
    } break;
  }
  ClearReceivedMessage();
}

void HandleSetRequest(SetRequest* set_request) {
  switch (set_request->which_type) {
    case SetRequest_output_gain_request_tag: {
      SetOutputGain(set_request->type.output_gain_request);
    } break;
    case SetRequest_denoising_request_tag: {
      SetDenoising(set_request->type.denoising_request);
    } break;
    case SetRequest_compression_request_tag: {
      SetCompression(set_request->type.compression_request);
    } break;
    case SetRequest_tactor_amplitudes_request_tag: {
      SetTactorAmplitudes(&(set_request->type.tactor_amplitudes_request));
    } break;
    case SetRequest_tactor_phases_request_tag: {
      SetTactorPhases(&(set_request->type.tactor_phases_request));
    } break;
    case SetRequest_tactor_frequencies_request_tag: {
      SetTactorFrequencies(&(set_request->type.tactor_frequencies_request));
    } break;
    default:
      break;
  }
}

void HandleGetRequest(GetRequest_GetOpion option) {
  switch (option) {
    case GetRequest_GetOpion_TACTOR_HARDWARE_LAYOUT: {
      GetTactorHardwareLayoutResponse();
    } break;
    case GetRequest_GetOpion_BATTERY: {
      GetBatteryResponse();
    } break;
    case GetRequest_GetOpion_SUPPORTED_OPERATION_MODES: {
      GetSupportedOperationModesResponse();
    } break;
    case GetRequest_GetOpion_CURRENT_OPERATION_MODE: {
      GetCurrentOperationModeResponse();
    } break;
    case GetRequest_GetOpion_OPERATION_IS_RUNNING: {
      GetOperationIsRunningResponse();
    } break;
    case GetRequest_GetOpion_OUTPUT_GAIN: {
      GetOutputGainResponse();
    } break;
    case GetRequest_GetOpion_DENOISING: {
      GetDenoisingResponse();
    } break;
    case GetRequest_GetOpion_COMPRESSION: {
      GetCompressionResponse();
    } break;
    case GetRequest_GetOpion_TACTOR_AMPLITUDES: {
      GetTactorAmplitudesResponse();
    } break;
    case GetRequest_GetOpion_TACTOR_PHASES: {
      GetTactorPhasesResponse();
    } break;
    case GetRequest_GetOpion_TACTOR_FREQUENCIES: {
      GetTactorFrequencies();
    } break;
    default: {
    } break;
  }
}
