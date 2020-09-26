/* Copyright 2019 Google LLC
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

#ifndef AUDIO_TO_TACTILE_TACTILE_PORTAUDIO_DEVICE_H_
#define AUDIO_TO_TACTILE_TACTILE_PORTAUDIO_DEVICE_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Finds the device index for a PortAudio device, specifying a name and minimum
 * number of input and output channels.
 *
 * Device lookup is done as follows:
 *
 * 1. If `name` can be parsed as integer, interpret it as a device index.
 *
 * 2. Otherwise, search for a device with name equal to `name`, ignoring case.
 *
 * 3. Otherwise, search for a device whose name contains `name`, ignoring case,
 *    and has at least the specified min numbers of input and output channels.
 *    [If there are multiple such devices, return the one with lowest index.]
 *
 * The function returns -1 if no match is found, or if the found device does not
 * have at least the specified min numbers of input and output channels.
 */
int FindPortAudioDevice(
    const char* name, int min_input_channels, int min_output_channels);

/* Prints a list of all PortAudio devices to stdout. */
void PrintPortAudioDevices();

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_TACTILE_PORTAUDIO_DEVICE_H_ */

