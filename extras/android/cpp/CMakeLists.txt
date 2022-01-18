# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.10.2)

project("audio_to_tactile")

add_library( # Name of the library.
             tuning_jni

             # Sets the library as a shared library.
             SHARED

             # Source files.
             tuning_jni.cpp
             ../../../src/dsp/fast_fun.c
             ../../../src/tactile/tuning.c)

include_directories(../../../src)

target_link_libraries(tuning_jni)
