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

#ifndef AUDIO_TO_TACTILE_SRC_NUS_MESSAGE_NUS_MESSAGE_H_
#define AUDIO_TO_TACTILE_SRC_NUS_MESSAGE_NUS_MESSAGE_H_

#include <pb_decode.h>
#include <pb_encode.h>

#include "app_config.h"      /* NOLINT(build/include) */
#include "tactile_data.pb.h" /* NOLINT(build/include) */

#ifdef __cplusplus
extern "C" {
#endif

void ClearSendMessage();
void ClearReceivedMessage();
bool EncodeTactileMessage();
void SendTactileMessage();

extern TactileMessage nus_send_message;
extern ClientMessage received_message;
/* extern pb_ostream_t send_write_stream; */

#ifdef __cplusplus
}
#endif

#endif  /* AUDIO_TO_TACTILE_SRC_NUS_MESSAGE_NUS_MESSAGE_H_ */
