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
 *
 * Initialize BLE stack.
 */

#ifndef AUDIO_TO_TACTILE_SRC_BLE_TACTILE_BLE_STACK_H_
#define AUDIO_TO_TACTILE_SRC_BLE_TACTILE_BLE_STACK_H_

#include <stdint.h>

#include "app_config.h"             /* NOLINT(build/include) */
#include "tactile_ble_connection.h" /* NOLINT(build/include) */

#ifdef __cplusplus
extern "C" {
#endif

#define kAppBleConnCfgTag 1
typedef void (*TactileBleStackCallback)(void);

/* Initialize Nordic BLE stack. Takes two callback functions that get executed
 * on the 'Connect' and 'disconnect' event.
 */
void BleStackInit(TactileBleStackCallback OnConnect,
                  TactileBleStackCallback OnDisconnect);

#ifdef __cplusplus
}
#endif

#endif  /* AUDIO_TO_TACTILE_SRC_BLE_TACTILE_BLE_STACK_H_ */
