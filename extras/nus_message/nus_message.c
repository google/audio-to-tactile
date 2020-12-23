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

#include "nus_message.h" /* NOLINT(build/include) */

#include "nrf_log.h" /* NOLINT(build/include) */

TactileMessage nus_send_message = TactileMessage_init_zero;
ClientMessage received_message = ClientMessage_init_zero;
static uint8_t nus_send_message_buffer[TactileMessage_size];
static pb_ostream_t send_write_stream;

void ClearSendMessage() {
  memset(&nus_send_message, 0, sizeof(TactileMessage));
}

void ClearReceivedMessage() {
  memset(&received_message, 0, sizeof(ClientMessage));
}

void SendTactileMessage() {
  NRF_LOG_DEBUG("send tactile message %d", send_write_stream.bytes_written);
  sendNusData(nus_send_message_buffer, send_write_stream.bytes_written);
}

bool EncodeTactileMessage() {
  send_write_stream = pb_ostream_from_buffer(nus_send_message_buffer,
                                             sizeof(nus_send_message_buffer));
  bool status =
      pb_encode(&send_write_stream, TactileMessage_fields, &nus_send_message);
  if (!status) {
    NRF_LOG_DEBUG("Encoding failed: %s\n", PB_GET_ERROR(&send_write_stream));
  }
  return status;
}
