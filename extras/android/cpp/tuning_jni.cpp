// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <jni.h>

#include <algorithm>
#include <cstdio>

#include "tactile/tuning.h"

#define JNI_METHOD(fun) \
  Java_com_google_audio_1to_1tactile_TuningNative_##fun  // NOLINT

// Gets the number of tuning knobs.
extern "C" JNIEXPORT jint JNICALL
JNI_METHOD(numTuningKnobs)(JNIEnv* env, jobject /* this */) {
  return kNumTuningKnobs;
}

// Gets the name string for `knob`.
extern "C" JNIEXPORT jstring JNICALL JNI_METHOD(name)(JNIEnv* env,
                                                      jobject /* this */,
                                                      jint knob) {
  return env->NewStringUTF(kTuningKnobInfo[knob].name);
}

// Gets the description string for `knob`.
extern "C" JNIEXPORT jstring JNICALL JNI_METHOD(description)(JNIEnv* env,
                                                             jobject /* this */,
                                                             jint knob) {
  return env->NewStringUTF(kTuningKnobInfo[knob].description);
}

// Gets the default value for `knob`.
extern "C" JNIEXPORT jint JNICALL JNI_METHOD(default)(JNIEnv* env,
                                                      jobject /* this */,
                                                      jint knob) {
  return static_cast<jint>(kDefaultTuningKnobs.values[knob]);
}

// Maps a knob control value and formats it as a string, including units if
// applicable.
extern "C" JNIEXPORT jstring JNICALL JNI_METHOD(mapControlValue)(
    JNIEnv* env, jobject /* this */, jint knob, jint value) {
  value = std::max(0, std::min(value, 255));
  const float mapped_value = ::TuningMapControlValue(knob, value);

  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), kTuningKnobInfo[knob].format,
                mapped_value);
  return env->NewStringUTF(buffer);
}
