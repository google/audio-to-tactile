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
# CARL+PCEN frontend for audio-to-tactile representation.

load("//:defs.bzl", "c_library", "c_test")

package(default_visibility = ["//visibility:public"])

c_library(
    name = "carl_frontend",
    srcs = ["carl_frontend.c"],
    hdrs = ["carl_frontend.h"],
    deps = [
        ":carl_frontend_design",
        "//dsp:fast_fun",
        "//dsp:math_constants",
    ],
)

c_library(
    name = "carl_frontend_design",
    srcs = ["carl_frontend_design.c"],
    hdrs = ["carl_frontend_design.h"],
    deps = [
        "//dsp:biquad_filter",
        "//dsp:complex",
        "//dsp:math_constants",
    ],
)

c_test(
    name = "carl_frontend_test",
    srcs = ["carl_frontend_test.c"],
    deps = [
        ":carl_frontend",
        ":carl_frontend_design",
        "//dsp:complex",
        "//dsp:logging",
        "//dsp:math_constants",
    ],
)
