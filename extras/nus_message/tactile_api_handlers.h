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

#ifndef AUDIO_TO_TACTILE_SRC_NUS_MESSAGE_TACTILE_API_HANDLES_H_
#define AUDIO_TO_TACTILE_SRC_NUS_MESSAGE_TACTILE_API_HANDLES_H_

#include "tactile_data.pb.h" /* NOLINT(build/include) */

#ifdef __cplusplus
extern "C" {
#endif

void GetTactorHardwareLayoutResponse();
void GetBatteryResponse();
void GetSupportedOperationModesResponse();
void GetCurrentOperationModeResponse();
void GetOperationIsRunningResponse();
void GetOutputGainResponse();
void GetDenoisingResponse();
void GetCompressionResponse();
void GetTactorAmplitudesResponse();
void GetTactorPhasesResponse();
void GetTactorFrequencies();

void SetTactorAmplitudes(TactorAmplitudes* tactor_amplitudes_p);
void SetTactorPhases(TactorPhases* tactor_phases_p);
void SetTactorFrequencies(TactorFrequencies* tactor_frequencies_p);
void SetOutputGain(int new_output_gain);
void SetDenoising(int new_denoising);
void SetCompression(int new_compression);

#ifdef __cplusplus
}
#endif

#endif  /* AUDIO_TO_TACTILE_SRC_NUS_MESSAGE_TACTILE_API_HANDLES_H_ */
