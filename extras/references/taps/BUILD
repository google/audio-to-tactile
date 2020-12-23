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
# Implementation of TAPS, TActile Phonemic Sleeve, of Reed et al.
#
# This directory implements a training game for TAPS. The system takes a
# sequence of phonemes as input and producing 24 channels of tactile signals as
# output, using distinct spatial and modulation patterns to encode different
# phonemes.
#
# Reference:
#   Charlotte M. Reed, Hong Z. Tan, Zachary D. Perez, E. Courtenay Wilson,
#   Frederico M. Severgnini, Jaehong Jung, Juan S. Martinez, Yang Jiao, Ali
#   Israr, Frances Lau, Keith Klumb, Robert Turcott, Freddy Abnousi, "A
#   Phonemic-Based Tactile Display for Speech Communication," IEEE
#   Transactions on Haptics, 2018.

load("//:defs.bzl", "c_binary", "c_library", "c_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

c_library(
    name = "phoneme_code",
    srcs = ["phoneme_code.c"],
    hdrs = ["phoneme_code.h"],
    deps = [
        "//:dsp",
        "//extras/tools:util",
    ],
)

c_test(
    name = "phoneme_code_test",
    srcs = ["phoneme_code_test.c"],
    deps = [
        ":phoneme_code",
        "//:dsp",
    ],
)

c_library(
    name = "tactile_player",
    srcs = ["tactile_player.c"],
    hdrs = ["tactile_player.h"],
    linkopts = ["-pthread"],
)

c_test(
    name = "tactile_player_test",
    srcs = ["tactile_player_test.c"],
    deps = [
        ":tactile_player",
        "//:dsp",
    ],
)

c_library(
    name = "tactophone",
    srcs = ["tactophone.c"],
    hdrs = ["tactophone.h"],
    deps = [
        ":phoneme_code",
        ":tactophone_engine",
        "//:dsp",
    ],
)

c_library(
    name = "tactophone_engine",
    srcs = ["tactophone_engine.c"],
    hdrs = ["tactophone_engine.h"],
    linkopts = [
        "-lncurses",
        "-lportaudio",
    ],
    deps = [
        ":phoneme_code",
        ":tactile_player",
        ":tactophone_lesson",
        "//:dsp",
        "//extras/tools:channel_map",
        "//extras/tools:util",
    ],
)

c_test(
    name = "tactophone_engine_test",
    srcs = ["tactophone_engine_test.c"],
    deps = [
        ":tactophone_engine",
        "//:dsp",
    ],
)

c_library(
    name = "tactophone_lesson",
    srcs = ["tactophone_lesson.c"],
    hdrs = ["tactophone_lesson.h"],
    deps = [
        "//:dsp",
    ],
)

c_test(
    name = "tactophone_lesson_test",
    srcs = ["tactophone_lesson_test.c"],
    deps = [
        ":tactophone_lesson",
        "//:dsp",
    ],
)

c_binary(
    name = "tactophone_main",
    srcs = ["tactophone_main.c"],
    deps = [
        ":tactophone",
        ":tactophone_states",
        "//:dsp",
        "//extras/tools:util",
    ],
)

c_library(
    name = "tactophone_states",
    srcs = [
        "tactophone_state_begin_lesson.c",
        "tactophone_state_free_play.c",
        "tactophone_state_lesson_done.c",
        "tactophone_state_lesson_review.c",
        "tactophone_state_lesson_trial.c",
        "tactophone_state_main_menu.c",
        "tactophone_state_test_tactors.c",
    ],
    hdrs = ["tactophone_states.h"],
    deps = [":tactophone_engine"],
)
