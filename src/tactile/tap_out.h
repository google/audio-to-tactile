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
 * When tap_out is capturing data, the device sends a "Capture" message of
 * binary data over USB serial to the PC or phone once every mic buffer. To
 * coexist with regular text-mode `Serial.println()` serial communication:
 *
 *  - tap_out only prints binary serial messages while a receiver is listening.
 *    If the device doesn't get a "heartbeat" message for ~400 ms, tap_out
 *    deactivates. This way text-mode messages can still be read in the Arduino
 *    serial monitor when tap_out is inactive.
 *
 *  - You should put `if (TapOutIsActive()) { ... }` around Serial.println()
 *    calls to avoid them from interrupting the binary communication. In case
 *    something slips by, the protocol uses marker bytes (0xfe) to ignore it.
 *
 * Instructions:
 *
 * 1. Use `TapOutSetTxFun()` and `TapOutSetErrorFun()` to set callbacks.
 *
 * 2. Use `TapOutAddDescriptor()` to define a "descriptor" with name and other
 *    metadata for each tap-out output that could be captured.
 *
 *    const TapOutDescriptor kMicInputDescriptor =
 *      {"mic input", "int16", 1, {kSamplesPerBuffer}};
 *    TapOutToken mic_input_token = TapOutAddDescriptor(&kMicInputDescriptor);
 *
 * 3. When serial data is received, call `TapOutReceiveMessage()` to parse it.
 *    This is used to respond to control messages, e.g. "Start Capture":
 *
 *    if (Serial.available() > 0) {
 *      char data[16];
 *      int size = min(Serial.available(), sizeof(data));
 *      Serial.readBytes(data, size);
 *      TapOutReceiveMessage(data, size);
 *    }
 *
 * 4. For each tap-out output, use `TapOutGetSlice()` to determine whether that
 *    output is currently enabled, and if it is, where to write the output.
 *
 *    const TapOutSlice* slice;
 *    if ((slice = TapOutGetSlice(mic_input_token))) {
 *      memcpy(slice->data, mic_samples_data, slice->size);
 *    }
 *
 * 5. At the end of each mic buffer, call `TapOutFinishedCaptureBuffer()`
 *    indicate that captured data should now be sent.
 *
 *    TapOutFinishedCaptureBuffer();
 *
 * On the wire, the protocol used is
 *
 *  [0] kTapOutMarker (= 0xfe)
 *  [1] <op code> - Indicates the type of message.
 *  [2] <payload size> - The size of the payload.
 *
 * Followed by the payload data.
 */

#ifndef AUDIO_TO_TACTILE_SRC_TACTILE_TAP_OUT_H_
#define AUDIO_TO_TACTILE_SRC_TACTILE_TAP_OUT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  /* Tap out buffer capacity in bytes. */
  kTapOutBufferCapacity = 256,
  /* Max number of dimensions in an output. */
  kTapOutMaxDims = 3,
  /* Max number of simultaneously enabled outputs. */
  kTapOutMaxOutputs = 4,
  /* Marker byte to help detect and skip across extra bytes. */
  kTapOutMarker = 0xfe,
};

/* Message ops for communication. */
enum {
  /* "Heartbeat" with empty payload to indicate that receiver is listening. */
  kTapOutMessageHeartbeat = 0x01,
  /* Request with empty payload to get the tap out descriptors. */
  kTapOutMessageGetDescriptors = 0x02,
  /* Message containing the descriptors. */
  kTapOutMessageDescriptors = 0x03,
  /* Request to begin capture. Payload specifies which outputs. */
  kTapOutMessageStartCapture = 0x04,
  /* Message containing captured tap out output. */
  kTapOutMessageCapture = 0x05,
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

/* Sets callback for transmitting serial data. */
void TapOutSetTxFun(void (*fun)(const char*, int));

/* Sets an error callback for printing error messages. */
void TapOutSetErrorFun(void (*fun)(const char*));

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

/* Returns 1 if tap out heartbeat is live, and 0 if not. To avoid conflict,
 * other serial communication should be avoided when this function returns 1.
 */
int /*bool*/ TapOutIsActive(void);

/* Interprets received serial data. */
void TapOutReceiveMessage(const char* data, int size);

/* Indicates that a buffer has just finished. */
void TapOutFinishedCaptureBuffer(void);

/* Enables `outputs` for capture. For each output, the function prepares a
 * TapOutSlice of g_tap_out_buffer to write data for that output. Returns 1 on
 * success, or 0 on failure (e.g. if buffer capacity is exceeded).
 *
 * All outputs may be disabled by passing NULL for `tokens`.
 */
int /*bool*/ TapOutEnable(const TapOutToken* outputs, int num_outputs);


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

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif  /* AUDIO_TO_TACTILE_SRC_TACTILE_TAP_OUT_H_ */
