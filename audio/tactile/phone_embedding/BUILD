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
# An embedding that is correlated with phones, useful to make distinguishable
# phones in audio-to-tactile representations.

load("//:defs.bzl", "c_binary", "c_library", "c_test")

package(default_visibility = ["//visibility:public"])

c_library(
    name = "embed_vowel",
    srcs = [
        "embed_vowel.c",
        "embed_vowel_params.h",
    ],
    hdrs = ["embed_vowel.h"],
    deps = [
        "//audio/dsp/portable:fast_fun",
        "//audio/tactile:hexagon_interpolation",
        "//audio/tactile:nn_ops",
    ],
)

c_test(
    name = "embed_vowel_test",
    srcs = ["embed_vowel_test.c"],
    data = [
        "//audio/tactile/testdata:phone_wavs",
    ],
    deps = [
        ":embed_vowel",
        "//audio/dsp/portable:logging",
        "//audio/dsp/portable:read_wav_file",
        "//audio/tactile/frontend:carl_frontend",
    ],
)

py_binary(
    name = "export_model_as_c_data",
    srcs = ["export_model_as_c_data.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":phone_model",
        "//audio/tactile/jax:hk_util",
    ],
)

py_binary(
    name = "ingest_timit",
    srcs = ["ingest_timit.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        "//audio/dsp/portable/python:wav_io",
        "//audio/tactile/python:read_nist_sphere",
    ],
)

py_binary(
    name = "make_training_data",
    srcs = ["make_training_data.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":phone_util",
        "//audio/dsp/portable/python:rational_factor_resampler",
        "//audio/dsp/portable/python:wav_io",
        "//audio/tactile/frontend/python:carl_frontend",
        "//audio/tactile/python:dsp",
    ],
)

py_library(
    name = "phone_util",
    srcs = ["phone_util.py"],
    deps = [
        "//audio/tactile/frontend/python:carl_frontend",
    ],
)

py_test(
    name = "phone_util_test",
    srcs = ["phone_util_test.py"],
    deps = [
        ":phone_util",
    ],
)

py_library(
    name = "phone_model",
    srcs = ["phone_model.py"],
    deps = [
        ":phone_util",
        "//audio/tactile/jax:hk_util",
        "//audio/tactile/phone_embedding/python:embed_vowel",
        "//audio/tactile/python:plot",
        "//audio/tactile/python:stats",
    ],
)

py_test(
    name = "phone_model_test",
    srcs = ["phone_model_test.py"],
    python_version = "PY3",
    deps = [
        ":phone_model",
        ":phone_util",
        "//audio/tactile/jax:hk_util",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    python_version = "PY3",
    deps = [
        ":phone_model",
    ],
)

py_binary(
    name = "run_phone_model_on_wav",
    srcs = ["run_phone_model_on_wav.py"],
    python_version = "PY3",
    deps = [
        ":phone_model",
        ":phone_util",
        "//audio/dsp/portable/python:wav_io",
        "//audio/tactile/frontend/python:carl_frontend",
        "//audio/tactile/jax:hk_util",
        "//audio/tactile/phone_embedding/python:embed_vowel",
        "//audio/tactile/python:dsp",
        "//audio/tactile/python:plot",
    ],
)
