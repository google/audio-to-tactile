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
#
#
# Tests for src/cpp.

package(licenses = ["notice"])

# Embedded development tools don't necessarily support the latest C++
# standard, so we set "-std=c++11" to check that code builds in C++11.
DEFAULT_COPTS = ["-std=c++11"]

cc_test(
    name = "message_test",
    srcs = ["message_test.cpp"],
    copts = DEFAULT_COPTS,
    deps = [
        "//:cpp",
        "//:dsp",
        "//:tactile",
        "//extras/tools:channel_map_tui",
    ],
)

cc_test(
    name = "object_pool_test",
    srcs = ["object_pool_test.cpp"],
    copts = DEFAULT_COPTS,
    deps = [
        "//:cpp",
        "//:dsp",
    ],
)

cc_test(
    name = "slice_test",
    srcs = ["slice_test.cpp"],
    copts = DEFAULT_COPTS,
    deps = [
        "//:cpp",
        "//:dsp",
    ],
)

cc_test(
    name = "std_shim_test",
    srcs = ["std_shim_test.cpp"],
    copts = DEFAULT_COPTS,
    deps = [
        "//:cpp",
        "//:dsp",
    ],
)
