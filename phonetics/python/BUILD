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
# Phonetics Python bindings.

load("//:defs.bzl", "py_extension")

package(default_visibility = ["//visibility:public"])
py_extension(
    name = "embed_vowel",
    srcs = ["embed_vowel_python_bindings.c"],
    deps = [
        "//phonetics:embed_vowel",
    ],
)

py_test(
    name = "embed_vowel_test",
    srcs = ["embed_vowel_test.py"],
    data = [
        "//testdata:phone_wavs",
    ],
    deps = [
        ":embed_vowel",
        "//dsp/python:wav_io",
        "//frontend/python:carl_frontend",
    ],
)

py_binary(
    name = "export_model_as_c_data",
    srcs = ["export_model_as_c_data.py"],
    python_version = "PY3",
    deps = [
        ":hk_util",
        ":phone_model",
    ],
)

py_library(
    name = "hk_util",
    srcs = ["hk_util.py"],
    srcs_version = "PY3",
)

py_test(
    name = "hk_util_test",
    srcs = ["hk_util_test.py"],
    srcs_version = "PY3",
    deps = [
        ":hk_util",
    ],
)

py_binary(
    name = "ingest_timit",
    srcs = ["ingest_timit.py"],
    python_version = "PY3",
    deps = [
        ":read_nist_sphere",
        "//dsp/python:wav_io",
    ],
)

py_binary(
    name = "make_training_data",
    srcs = ["make_training_data.py"],
    python_version = "PY3",
    deps = [
        ":phone_util",
        ":sliding_window",
        "//dsp/python:rational_factor_resampler",
        "//dsp/python:wav_io",
        "//frontend/python:carl_frontend",
    ],
)

py_library(
    name = "phone_util",
    srcs = ["phone_util.py"],
    deps = [
        "//frontend/python:carl_frontend",
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
        ":embed_vowel",
        ":hk_util",
        ":phone_util",
        ":plot",
        ":stats",
    ],
)

py_test(
    name = "phone_model_test",
    srcs = ["phone_model_test.py"],
    python_version = "PY3",
    deps = [
        ":hk_util",
        ":phone_model",
        ":phone_util",
    ],
)

py_library(
    name = "plot",
    srcs = ["plot.py"],
    srcs_version = "PY3",
)

py_test(
    name = "plot_test",
    srcs = ["plot_test.py"],
    deps = [
        ":plot",
    ],
)

py_library(
    name = "read_nist_sphere",
    srcs = ["read_nist_sphere.py"],
)

py_test(
    name = "read_nist_sphere_test",
    srcs = ["read_nist_sphere_test.py"],
    deps = [
        ":read_nist_sphere",
    ],
)

py_binary(
    name = "run_phone_model_on_wav",
    srcs = ["run_phone_model_on_wav.py"],
    python_version = "PY3",
    deps = [
        ":embed_vowel",
        ":hk_util",
        ":phone_model",
        ":phone_util",
        ":plot",
        ":sliding_window",
        "//dsp/python:wav_io",
        "//frontend/python:carl_frontend",
    ],
)

py_library(
    name = "sliding_window",
    srcs = ["sliding_window.py"],
)

py_test(
    name = "sliding_window_test",
    srcs = ["sliding_window_test.py"],
    deps = [
        ":sliding_window",
    ],
)

py_library(
    name = "stats",
    srcs = ["stats.py"],
)

py_test(
    name = "stats_test",
    srcs = ["stats_test.py"],
    deps = [
        ":stats",
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
