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
 * Set up Bluetooth Generic Access Profile (GAP).
 */

#ifndef AUDIO_TO_TACTILE_SRC_BLE_TACTILE_BLE_GAP_H_
#define AUDIO_TO_TACTILE_SRC_BLE_TACTILE_BLE_GAP_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifndef kDeviceName
#define kDeviceName "AudioToTactile"
#endif

/* Gap parameter initialization. */
void GapParamsInit();

#ifdef __cplusplus
}
#endif

#endif  /* AUDIO_TO_TACTILE_SRC_BLE_TACTILE_BLE_GAP_H_ */
