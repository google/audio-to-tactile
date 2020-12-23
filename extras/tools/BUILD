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
# Tools and non-embedded code.

load("//:defs.bzl", "c_binary", "c_library", "c_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

c_library(
    name = "channel_map",
    srcs = ["channel_map.c"],
    hdrs = ["channel_map.h"],
    deps = [
        ":util",
    ],
)

c_test(
    name = "channel_map_test",
    srcs = ["channel_map_test.c"],
    deps = [
        ":channel_map",
        ":util",
        "//:dsp",
    ],
)

c_binary(
    name = "play_buzz",
    srcs = ["play_buzz.c"],
    deps = [
        ":portaudio_device",
        ":util",
        "//:dsp",
    ],
)

c_library(
    name = "portaudio_device",
    srcs = ["portaudio_device.c"],
    hdrs = ["portaudio_device.h"],
    linkopts = ["-lportaudio"],
    deps = [
        ":util",
    ],
)

c_binary(
    name = "run_energy_envelope",
    srcs = ["run_energy_envelope.c"],
    deps = [
        ":portaudio_device",
        ":util",
        "//:dsp",
        "//:tactile",
    ],
)

c_binary(
    name = "run_energy_envelope_on_wav",
    srcs = ["run_energy_envelope_on_wav.c"],
    deps = [
        ":util",
        "//:dsp",
        "//:tactile",
    ],
)

c_binary(
    name = "run_demuxer_on_wav",
    srcs = ["run_demuxer_on_wav.c"],
    deps = [
        ":util",
        "//:dsp",
        "//:mux",
    ],
)

c_binary(
    name = "run_muxer_on_wav",
    srcs = ["run_muxer_on_wav.c"],
    deps = [
        ":util",
        "//:dsp",
        "//:mux",
    ],
)

c_binary(
    name = "run_tactile_methods",
    srcs = ["run_tactile_methods.c"],
    deps = [
        ":channel_map",
        ":portaudio_device",
        ":util",
        "//:dsp",
        "//:tactile",
        "//extras/references/bratakos2001",
        "//extras/references/yuan2005",
    ],
)

c_binary(
    name = "run_tactile_processor",
    srcs = ["run_tactile_processor.c"],
    deps = [
        ":channel_map",
        ":portaudio_device",
        ":run_tactile_processor_assets",
        ":util",
        "//:dsp",
        "//:tactile",
        "//extras/tools/sdl:basic_sdl_app",
        "//extras/tools/sdl:texture_from_rle_data",
        "//extras/tools/sdl:window_icon",
        "@sdl2",
    ],
)

c_library(
    name = "run_tactile_processor_assets",
    srcs = ["run_tactile_processor_assets.c"],
)

c_binary(
    name = "tactometer",
    srcs = ["tactometer.c"],
    deps = [
        ":portaudio_device",
        ":util",
        "//:dsp",
        "//extras/tools/sdl:basic_sdl_app",
        "//extras/tools/sdl:draw_text",
        "//extras/tools/sdl:window_icon",
        "@sdl2",
    ],
)

c_library(
    name = "util",
    srcs = ["util.c"],
    hdrs = ["util.h"],
    deps = [
        "//:dsp",
    ],
)

c_test(
    name = "util_test",
    srcs = ["util_test.c"],
    deps = [
        ":util",
        "//:dsp",
    ],
)
