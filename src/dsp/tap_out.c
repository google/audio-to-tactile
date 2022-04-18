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
 */

#include "dsp/tap_out.h"

#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

enum {
  /* Max number of output descriptors that can be defined. */
  kMaxDescriptors = 12,
  /* Max name length before it is truncated in the binary format. */
  kMaxNameLength = 15,
  /* Number of bytes per descriptor in `TapOutWriteDescriptors()`. */
  kBytesPerDescriptor = 1 + kMaxNameLength + 1 + kTapOutMaxDims,
  /* Max number of buffers to send without receiving another heartbeat. Set such
   * that the receiver can send once every 100 buffers plus a little slack.
   */
  kMaxBuffersPerHeartbeat = 100 + 5,
};

typedef enum {
  kDTypeInvalid,
  kDTypeUint8,
  kDTypeInt8,
  kDTypeUint16,
  kDTypeInt16,
  kDTypeUint32,
  kDTypeInt32,
  kDTypeUint64,
  kDTypeInt64,
  kDTypeFloat,
  kDTypeDouble,
  kDTypeText,
} DType;

typedef uint8_t TapOutToken;

uint8_t g_tap_out_buffer[kTapOutBufferCapacity];
int g_tap_out_buffer_size = 0;

static const TapOutDescriptor* g_descriptors[kMaxDescriptors];
static int g_num_descriptors = 0;

static TapOutToken g_outputs[kTapOutMaxOutputs];
static TapOutSlice g_slices[kTapOutMaxOutputs];
static int g_num_outputs = 0;
static int g_heartbeat_countdown = 0;

static void (*g_tx_fun)(const char*, int) = NULL;
static void (*g_error_fun)(const char*) = NULL;

/* Prints a formatted error message with g_error_fun, if set. */
static void Error(const char* format, ...) {
  if (g_error_fun) {
    va_list args;
    va_start(args, format);
    char buffer[64];
    vsprintf(buffer, format, args);
    va_end(args);

    g_error_fun(buffer);
  }
}

/* Parses a dtype string. Returns kDTypeInvalid on failure. */
static DType DTypeParse(const char* dtype) {
  if (dtype == NULL) { return kDTypeInvalid; }
  switch (dtype[0]) {
    case 'u':
      if (!strcmp(dtype, "uint8")) { return kDTypeUint8; }
      if (!strcmp(dtype, "uint16")) { return kDTypeUint16; }
      if (!strcmp(dtype, "uint32")) { return kDTypeUint32; }
      if (!strcmp(dtype, "uint64")) { return kDTypeUint64; }
      break;

    case 'i':
      if (!strcmp(dtype, "int8")) { return kDTypeInt8; }
      if (!strcmp(dtype, "int16")) { return kDTypeInt16; }
      if (!strcmp(dtype, "int32")) { return kDTypeInt32; }
      if (!strcmp(dtype, "int64")) { return kDTypeInt64; }
      break;

    default:
      if (!strcmp(dtype, "float")) { return kDTypeFloat; }
      if (!strcmp(dtype, "double")) { return kDTypeDouble; }
      if (!strcmp(dtype, "text")) { return kDTypeText; }
  }
  return kDTypeInvalid;
}

/* Gets the number of bytes per item for `dtype`. */
static int DTypeItemSize(DType dtype) {
  static const int kNumBytes[] = {0, 1, 1, 2, 2, 4, 4, 8, 8, 4, 8, 1};
  return kNumBytes[dtype];
}

/* Computes the number of bytes from dtype and shape. */
static int ComputeNumBytes(const TapOutDescriptor* descriptor) {
  if (!(1 <= descriptor->num_dims && descriptor->num_dims <= kTapOutMaxDims)) {
    Error("%s: Invalid num_dims: %d", descriptor->name, descriptor->num_dims);
    return 0;
  }
  int num_bytes = DTypeItemSize(DTypeParse(descriptor->dtype));
  if (num_bytes <= 0) {
    Error("%s: Invalid dtype: \"%s\"", descriptor->name, descriptor->dtype);
    return 0;
  }

  int i;
  for (i = 0; i < descriptor->num_dims; ++i) {
    const int dim = descriptor->shape[i];
    if (dim < 1) {
      Error("%s: Invalid dim: %d", descriptor->name, dim);
      return 0;
    } else if (dim > kTapOutBufferCapacity) {
      Error("%s: Dim exceeds kTapOutBufferCapacity: %d", descriptor->name, dim);
      return 0;
    }
    num_bytes *= dim;
    if (num_bytes > kTapOutBufferCapacity) {
      Error("%s: Num bytes exceeds kTapOutBufferCapacity.", descriptor->name);
      return 0;
    }
  }

  return num_bytes;
}

void TapOutSetTxFun(void (*fun)(const char*, int)) {
  g_tx_fun = fun;
}

void TapOutSetErrorFun(void (*fun)(const char*)) {
  g_error_fun = fun;
}

TapOutToken TapOutAddDescriptor(const TapOutDescriptor* descriptor) {
  /* Validate descriptor fields. */
  if (descriptor == NULL) {
    Error("Null descriptor.");
    return 0;
  } else if (descriptor->name == NULL) {
    Error("Descriptor name field is null.");
    return 0;
  } else if (g_num_descriptors >= kMaxDescriptors) {
    Error("%s: Exceeded kMaxDescriptors (%d).",
          descriptor->name, kMaxDescriptors);
    return 0;
  } else if (ComputeNumBytes(descriptor) <= 0) {
    /* ComputeNumBytes() printed an error message if we reach here. */
    return 0;
  }

  g_descriptors[g_num_descriptors] = descriptor;
  return ++g_num_descriptors;
}

int TapOutWriteDescriptors(void) {

  const int total = 4 + g_num_descriptors * kBytesPerDescriptor;
  if (total > kTapOutBufferCapacity) {
    Error("Descriptors exceed kTapOutBufferCapacity.");
    return 0;
  }

  g_tap_out_buffer_size = total;
  memset(g_tap_out_buffer, 0, g_tap_out_buffer_size);
  uint8_t* dest = g_tap_out_buffer;

  *dest++ = kTapOutMarker;
  *dest++ = kTapOutMessageDescriptors;
  *dest++ = g_num_descriptors * kBytesPerDescriptor;
  *dest++ = (uint8_t)g_num_descriptors;

  int i;
  for (i = 0; i < g_num_descriptors; ++i) {
    const TapOutDescriptor* descriptor = g_descriptors[i];
    const TapOutToken token = i + 1;
    *dest++ = token;

    int name_len = strlen(descriptor->name);
    if (name_len > kMaxNameLength) { name_len = kMaxNameLength; }
    memcpy(dest, descriptor->name, name_len);
    dest += kMaxNameLength;

    *dest++ = DTypeParse(descriptor->dtype);
    int j;
    for (j = 0; j < descriptor->num_dims; ++j) {
      dest[j] = (uint8_t)descriptor->shape[j];
    }
    dest += kTapOutMaxDims;
  }

  return 1;
}

void TapOutClearDescriptors(void) {
  g_num_descriptors = 0;
}

int TapOutEnable(const TapOutToken* outputs, int num_outputs) {
  g_tap_out_buffer_size = 0;
  g_num_outputs = 0;
  if (outputs == NULL || num_outputs <= 0) {
    return 1;
  }

  if (num_outputs >= kTapOutMaxOutputs) {
    Error("TapOutEnable: Exceeded kTapOutMaxOutputs (%d).", num_outputs);
    return 0;
  }

  int offset = 3;
  g_tap_out_buffer[0] = kTapOutMarker;
  g_tap_out_buffer[1] = kTapOutMessageCapture;

  int i;
  for (i = 0; i < num_outputs; ++i) {
    if (!(1 <= outputs[i] && outputs[i] <= g_num_descriptors)) { return 0; }
    const int num_bytes = ComputeNumBytes(g_descriptors[outputs[i] - 1]);
    if (num_bytes <= 0) { return 0; }

    g_outputs[i] = outputs[i];
    g_slices[i].data = g_tap_out_buffer + offset;
    g_slices[i].size = num_bytes;

    offset += num_bytes;
    if (offset > kTapOutBufferCapacity) {
      Error("Total size exceeds kTapOutBufferCapacity.");
      return 0;
    }
  }

  g_tap_out_buffer_size = offset;
  g_tap_out_buffer[2] = offset - 3;  /* Set payload size. */
  g_num_outputs = num_outputs;
  return 1;
}

int TapOutIsActive(void) {
  return g_heartbeat_countdown > 0;
}

const TapOutSlice* TapOutGetSlice(TapOutToken output) {
  if (g_num_outputs <= 0) { return NULL; }
  const TapOutToken* p =
      (const TapOutToken*)memchr(g_outputs, output, g_num_outputs);
  if (p == NULL) { return NULL; }
  return &g_slices[(int)(p - g_outputs)];
}

void TapOutTextPrint(TapOutToken output, const char* format, ...) {
  const TapOutSlice* slice = TapOutGetSlice(output);
  if (slice) {
    memset(slice->data, 0, slice->size);
    va_list args;
    va_start(args, format);
    vsprintf((char*)slice->data, format, args);
    va_end(args);
  }
}

/* Calls `g_tx_fun` on the tap_out buffer if it is set. */
static void SendBuffer(void) {
  if (g_tx_fun) {
    g_tx_fun((const char*)g_tap_out_buffer, g_tap_out_buffer_size);
  }
}

/* Handles a received message. */
static void HandleMessage(int op, const uint8_t* payload, int payload_size) {
  g_heartbeat_countdown = kMaxBuffersPerHeartbeat; /* Reset the countdown. */

  switch (op) {
    case kTapOutMessageHeartbeat:
      break;

    case kTapOutMessageGetDescriptors:
      if (payload_size == 0 && TapOutWriteDescriptors()) {
        SendBuffer();
      }
      break;

    case kTapOutMessageStartCapture:
      if (1 <= payload_size && payload_size <= kTapOutMaxOutputs) {
        TapOutEnable(payload, payload_size);
      }
      break;

    default:
      Error("Unknown op: 0x%02x", op);
  }
}

void TapOutReceiveMessage(const char* data, int size) {
  /* The message structure is "marker, op, payload_size" followed by payload. */
  const char* marker = memchr(data, kTapOutMarker, size); /* Find the marker. */
  if (marker == NULL) { return; }
  ++marker;
  size -= (int)(marker - data);
  if (size < 2) { return; }
  data = marker;

  /* Read the op and payload size. */
  const int op = data[0];
  const int payload_size = data[1];
  if (size < payload_size) { return; }
  const uint8_t* payload = (const uint8_t*)(data + 2);

  HandleMessage(op, payload, payload_size);
}

void TapOutFinishedCaptureBuffer(void) {
  if (!g_heartbeat_countdown) { return; }

  --g_heartbeat_countdown;
  if (g_heartbeat_countdown == 0) {
    TapOutEnable(NULL, 0); /* Deactivate tap_out. */
  } else if (g_num_outputs > 0) {
    SendBuffer();
  }
}
