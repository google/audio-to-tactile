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
 *
 * Set up the Nordic UART Service (NUS) and supply proto parsing function.
 */

#ifndef AUDIO_TO_TACTILE_SRC_BLE_TACTILE_BLE_SERVICE_NUS_H_
#define AUDIO_TO_TACTILE_SRC_BLE_TACTILE_BLE_SERVICE_NUS_H_

#include <stdint.h>

#include "app_config.h"             /* NOLINT(build/include) */
#include "sdk_config.h"             /* NOLINT(build/include) */
#include "tactile_ble_connection.h" /* NOLINT(build/include) */

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ParseNusMessageToProto)(uint8_t* buffer, uint16_t length);
/* Nordic UART service initialization. */
void TactileBleServiceNusInit(ParseNusMessageToProto nus_message_to_proto);
void SendNusData(uint8_t* buffer, uint16_t length);

extern uint8_t g_ble_receive_buffer[NRF_SDH_BLE_GATT_MAX_MTU_SIZE];

#ifdef __cplusplus
}
#endif

#endif  /* AUDIO_TO_TACTILE_SRC_BLE_TACTILE_BLE_SERVICE_NUS_H_ */
