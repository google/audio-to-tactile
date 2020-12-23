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
# Utilities for working with the Simple DirectMedia Layer (SDL) library.

load("//:defs.bzl", "c_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

c_library(
    name = "basic_sdl_app",
    srcs = ["basic_sdl_app.c"],
    hdrs = ["basic_sdl_app.h"],
    deps = [
        "@sdl2",
    ],
)

c_library(
    name = "draw_text",
    srcs = ["draw_text.c"],
    hdrs = ["draw_text.h"],
    deps = [
        "@sdl2",
    ],
)

c_library(
    name = "texture_from_rle_data",
    srcs = ["texture_from_rle_data.c"],
    hdrs = ["texture_from_rle_data.h"],
    deps = [
        "//:dsp",
        "@sdl2",
    ],
)

c_library(
    name = "window_icon",
    srcs = ["window_icon.c"],
    hdrs = ["window_icon.h"],
    deps = [
        "@sdl2",
    ],
)
