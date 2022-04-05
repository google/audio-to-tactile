/* Copyright 2022 Google LLC
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
 * A library to "tap out" intermediate outputs for analysis and debugging.
 *
 * It is useful when debugging to capture intermediate signals, for instance
 * the microphone input or Enveloper noise estimates. The approach with this
 * library a set of tap-out outputs is first defined, then any (and possibly
 * multiple) outputs may be selected for capture, and this selection may be
 * changed dynamically.
 *
 * 1. Use `TapOutAddDescriptor()` to define a "descriptor" with name and other
 *    metadata for each tap-out output that could be captured.
 *
 *    const TapOutDescriptor kMicInputDescriptor =
 *      {"mic input", "i2", 1, {kSamplesPerBuffer}};
 *    TapOutToken mic_input_token = TapOutAddDescriptor(&kMicInputDescriptor);
 *
 * 2. Use `TapOutWriteDescriptors()` to serialize the descriptor metadata. This
 *    serialization tells the receiver what outputs are available.
 *
 *    if (TapOutWriteDescriptors()) {
 *      Serial.write((const char*)g_tap_out_buffer, g_tap_out_buffer_size);
 *    }
 *
 * 3. Select which outputs to capture with `TapOutEnable()`. This may be done
 *    at any time so that which outputs are selected may be changed dynamically.
 *
 *    TapOutTokens outputs[kTapOutMaxOutputs];
 *    outputs[0] = mic_input_token;
 *    TapOutEnable(outputs, 1);
 *
 * 4. For each tap-out output, use `TapOutGetSlice()` to determine whether that
 *    output is currently enabled, and if it is, where to write the output.
 *
 *    const TapOutSlice* slice;
 *    if ((slice = TapOutGetSlice(mic_input_token))) {
 *      memcpy(slice->data, mic_samples_data, slice->size);
 *    }
 *
 * 5. If any outputs are enabled, the buffer should be sent to the receiver
 *    after each time it is filled.
 *
 *    if (TapOutIsEnabled()) {
 *      Serial.write((const char*)g_tap_out_buffer, g_tap_out_buffer_size);
 *    }
 */

#ifndef AUDIO_TO_TACTILE_SRC_DSP_TAP_OUT_H_
#define AUDIO_TO_TACTILE_SRC_DSP_TAP_OUT_H_

#include <stdint.h>

enum {
  /* Tap out buffer capacity in bytes. */
  kTapOutBufferCapacity = 256,
  /* Max number of dimensions in an output. */
  kTapOutMaxDims = 3,
  /* Max number of simultaneously enabled outputs. */
  kTapOutMaxOutputs = 4,
  /* Marker byte to help the receiver detect and skip across extra bytes. */
  kTapOutMarker = 0xfe,
};

/* Descriptor metadata for one tap-out output. */
typedef struct {
  /* Human-readable name for this output. */
  const char* name;
  /* Element data type. Data is expected to be written in little endian order.
   *
   *   dtype     Meaning
   *   "uint8"   Unsigned 8-bit.
   *   "int8"    Signed 8-bit.
   *   "uint16"  Unsigned 16-bit.
   *   "int16"   Signed 16-bit.
   *   "uint32"  Unsigned 32-bit.
   *   "int32"   Signed 32-bit.
   *   "uint64"  Unsigned 64-bit.
   *   "int64"   Signed 64-bit.
   *   "float"   32-bit single-precision float.
   *   "double"  64-bit double-precision float.
   *   "text"    ASCII text data.
   */
  const char* dtype;
  /* Number of dimensions between 1 and kTapOutMaxDims. */
  int num_dims;
  /* Data shape in row major order. */
  int shape[kTapOutMaxDims];
} TapOutDescriptor;

typedef struct {
  uint8_t* data;
  int size;
} TapOutSlice;

/* Tokens are used as handles to refer to tap-out outputs. */
typedef uint8_t TapOutToken;
enum { kInvalidTapOutToken = 0 };

/* Buffer where tap out data is written. */
extern uint8_t g_tap_out_buffer[kTapOutBufferCapacity];
extern int g_tap_out_buffer_size;

/* Adds a descriptor for one tap-out output and returns a token to refer to this
 * output. On failure, a zero-valued token is returned. Additionally, a
 * diagnostic message is printed to the error function if one was set with
 * `TapOutSetErrorFun()`.
 */
TapOutToken TapOutAddDescriptor(const TapOutDescriptor* descriptor);

/* Serializes descriptor metadata to g_tap_out_buffer in a simple binary format.
 * The device should send this to the receiving end to inform it what tap-out
 * outputs are available, what their tokens are for `TapOutEnable()`, and how to
 * interpret the data. This is useful so that tap-out outputs can be changed in
 * the firmware without needing to update the receiver implementation.
 */
int /*bool*/ TapOutWriteDescriptors(void);

/* Clears all descriptors. */
void TapOutClearDescriptors(void);

/* Enables `outputs` for capture. For each output, the function prepares a
 * TapOutSlice of g_tap_out_buffer to write data for that output. Returns 1 on
 * success, or 0 on failure (e.g. if buffer capacity is exceeded).
 *
 * All outputs may be disabled by passing NULL for `tokens`.
 */
int /*bool*/ TapOutEnable(const TapOutToken* outputs, int num_outputs);

/* Returns 1 if any outputs are enabled, and 0 if not. */
int /*bool*/ TapOutIsEnabled(void);

/* Gets the buffer slice associated with `output`, if it is enabled, or returns
 * NULL if that output is disabled.
 */
const TapOutSlice* TapOutGetSlice(TapOutToken output);

/* For a text output, if enabled, print formatted text into the slice. Excess
 * bytes are filled with zeros.
 * NOTE: The caller must ensure the formatted string including null terminator
 * fit within the slice size.
 */
void TapOutTextPrint(TapOutToken output, const char* format, ...);

/* Sets an error callback for printing error messages. */
void TapOutSetErrorFun(void (*fun)(const char*));

#endif  /* AUDIO_TO_TACTILE_SRC_DSP_TAP_OUT_H_ */
