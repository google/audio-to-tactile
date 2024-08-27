# Copyright 2020 Google LLC
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
# Tests for src/mux.

load("//:defs.bzl", "c_test")

licenses(["notice"])

c_test(
    name = "muxer_test",
    srcs = ["muxer_test.c"],
    deps = [
        "//:dsp",
        "//:mux",
    ],
)

c_test(
    name = "pilot_tracker_test",
    srcs = ["pilot_tracker_test.c"],
    deps = [
        "//:dsp",
        "//:mux",
    ],
)
