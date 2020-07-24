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

load("//:defs.bzl", "c_binary", "c_library", "c_test")

package(default_visibility = ["//visibility:public"])

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
        "//audio/dsp/portable:logging",
    ],
)

c_library(
    name = "hexagon_interpolation",
    srcs = ["hexagon_interpolation.c"],
    hdrs = ["hexagon_interpolation.h"],
)

c_test(
    name = "hexagon_interpolation_test",
    srcs = ["hexagon_interpolation_test.c"],
    deps = [
        ":hexagon_interpolation",
        "//audio/dsp/portable:logging",
        "//audio/dsp/portable:math_constants",
    ],
)

c_library(
    name = "nn_ops",
    srcs = ["nn_ops.c"],
    hdrs = ["nn_ops.h"],
)

c_test(
    name = "nn_ops_test",
    srcs = ["nn_ops_test.c"],
    deps = [
        ":nn_ops",
        "//audio/dsp/portable:logging",
    ],
)

c_binary(
    name = "play_buzz",
    srcs = ["play_buzz.c"],
    linkopts = [
        "-lncurses",
        "-lportaudio",
    ],
    deps = [
        ":util",
    ],
)

c_binary(
    name = "play_wav",
    srcs = ["play_wav.c"],
    linkopts = [
        "-lncurses",
    ],
    deps = [
        ":channel_map",
        ":portaudio_device",
        ":util",
        "//audio/dsp/portable:fast_fun",
        "//audio/dsp/portable:number_util",
        "//audio/dsp/portable:read_wav_file",
    ],
)

c_library(
    name = "portaudio_device",
    srcs = ["portaudio_device.c"],
    hdrs = ["portaudio_device.h"],
    linkopts = [
        "-lportaudio",
    ],
    deps = [
        ":util",
    ],
)

c_library(
    name = "post_processor",
    srcs = ["post_processor.c"],
    hdrs = ["post_processor.h"],
    deps = [
        ":tactor_equalizer",
        "//audio/dsp/portable:biquad_filter",
        "//audio/dsp/portable:butterworth",
    ],
)

c_binary(
    name = "run_tactile_methods",
    srcs = ["run_tactile_methods.c"],
    deps = [
        ":channel_map",
        ":portaudio_device",
        ":util",
        "//audio/dsp/portable:auto_gain_control",
        "//audio/tactile/energy_envelope",
        "//audio/tactile/references/bratakos2001",
        "//audio/tactile/references/yuan2005",
    ],
)

c_binary(
    name = "run_tactile_processor",
    srcs = ["run_tactile_processor.c"],
    deps = [
        ":channel_map",
        ":portaudio_device",
        ":post_processor",
        ":run_tactile_processor_assets",
        ":tactile_processor",
        ":util",
        "//audio/dsp/portable:fast_fun",
        "//audio/dsp/portable:math_constants",
        "//audio/dsp/portable:number_util",
        "//audio/dsp/portable:read_wav_file",
        "//audio/tactile/sdl:basic_sdl_app",
        "//audio/tactile/sdl:texture_from_rle_data",
        "//audio/tactile/sdl:window_icon",
    ],
)

c_library(
    name = "run_tactile_processor_assets",
    srcs = ["run_tactile_processor_assets.c"],
)

c_library(
    name = "tactile_processor",
    srcs = ["tactile_processor.c"],
    hdrs = ["tactile_processor.h"],
    deps = [
        ":hexagon_interpolation",
        "//audio/tactile/energy_envelope",
        "//audio/tactile/frontend:carl_frontend",
        "//audio/tactile/phone_embedding:embed_vowel",
    ],
)

c_test(
    name = "tactile_processor_test",
    srcs = ["tactile_processor_test.c"],
    data = [
        "//audio/tactile/testdata:phone_wavs",
    ],
    deps = [
        ":tactile_processor",
        "//audio/dsp/portable:logging",
        "//audio/dsp/portable:math_constants",
        "//audio/dsp/portable:read_wav_file",
    ],
)

c_binary(
    name = "tactometer",
    srcs = ["tactometer.c"],
    deps = [
        ":portaudio_device",
        ":util",
        "//audio/dsp/portable:logging",
        "//audio/dsp/portable:math_constants",
        "//audio/tactile/sdl:basic_sdl_app",
        "//audio/tactile/sdl:draw_text",
        "//audio/tactile/sdl:window_icon",
    ],
)

c_library(
    name = "tactor_equalizer",
    srcs = ["tactor_equalizer.c"],
    hdrs = ["tactor_equalizer.h"],
    deps = [
        "//audio/dsp/portable:biquad_filter",
        "//audio/dsp/portable:math_constants",
    ],
)

c_test(
    name = "tactor_equalizer_test",
    srcs = ["tactor_equalizer_test.c"],
    deps = [
        ":tactor_equalizer",
        ":util",
        "//audio/dsp/portable:complex",
        "//audio/dsp/portable:logging",
        "//audio/dsp/portable:math_constants",
    ],
)

c_library(
    name = "util",
    srcs = ["util.c"],
    hdrs = ["util.h"],
    deps = [
        "//audio/dsp/portable:math_constants",
    ],
)

c_test(
    name = "util_test",
    srcs = ["util_test.c"],
    deps = [
        ":util",
        "//audio/dsp/portable:logging",
    ],
)
