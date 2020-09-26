# Copyright 2019 Google LLC
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
#
#
# Single-band envelope tactile feature of Bratakos et al.

load("//:defs.bzl", "c_library", "c_test")

package(default_visibility = ["//visibility:public"])

c_library(
    name = "bratakos2001",
    srcs = ["bratakos2001.c"],
    hdrs = ["bratakos2001.h"],
    deps = [
        "//dsp:biquad_filter",
        "//dsp:butterworth",
        "//dsp:math_constants",
        "//dsp:phasor_rotator",
    ],
)

c_test(
    name = "bratakos2001_test",
    srcs = ["bratakos2001_test.c"],
    deps = [
        ":bratakos2001",
        "//dsp:logging",
        "//dsp:math_constants",
    ],
)

