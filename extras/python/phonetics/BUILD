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
# Python code for phonetic network training.

load("//:defs.bzl", "py_extension")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_extension(
    name = "classify_phoneme",
    srcs = ["classify_phoneme_python_bindings.c"],
    deps = [
        "//:phonetics",
    ],
)

py_test(
    name = "classify_phoneme_test",
    srcs = ["classify_phoneme_test.py"],
    data = ["//extras/test/testdata:phone_wavs"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":classify_phoneme",
        "//extras/python:dsp",
        "//extras/python:frontend",
        "//extras/python/phonetics:sliding_window",
    ],
)

py_extension(
    name = "embed_vowel",
    srcs = ["embed_vowel_python_bindings.c"],
    deps = [
        "//:phonetics",
    ],
)

py_test(
    name = "embed_vowel_test",
    srcs = ["embed_vowel_test.py"],
    data = ["//extras/test/testdata:phone_wavs"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":embed_vowel",
        "//extras/python:dsp",
        "//extras/python:frontend",
    ],
)

py_binary(
    name = "export_model_as_c_data",
    srcs = ["export_model_as_c_data.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":hk_util",
        ":phone_model",
    ],
)

py_library(
    name = "hk_util",
    srcs = ["hk_util.py"],
    srcs_version = "PY3",
    deps = [
    ],
)

py_test(
    name = "hk_util_test",
    size = "medium",
    srcs = ["hk_util_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["noasan"],
    deps = [
        ":hk_util",
    ],
)

py_binary(
    name = "ingest_timit",
    srcs = ["ingest_timit.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":read_nist_sphere",
        "//extras/python:dsp",
    ],
)

py_binary(
    name = "make_training_data",
    srcs = ["make_training_data.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":phone_util",
        ":sliding_window",
        "//extras/python:dsp",
        "//extras/python:frontend",
    ],
)

py_library(
    name = "phone_model",
    srcs = ["phone_model.py"],
    srcs_version = "PY3",
    deps = [
        ":hk_util",
        ":phone_util",
        ":stats",
        "//extras/python:plot",
    ],
)

py_test(
    name = "phone_model_test",
    size = "medium",
    srcs = ["phone_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":hk_util",
        ":phone_model",
        ":phone_util",
    ],
)

py_library(
    name = "phone_util",
    srcs = ["phone_util.py"],
    srcs_version = "PY3",
    deps = [
        "//extras/python:frontend",
    ],
)

py_test(
    name = "phone_util_test",
    srcs = ["phone_util_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":phone_util",
    ],
)

py_library(
    name = "read_nist_sphere",
    srcs = ["read_nist_sphere.py"],
    srcs_version = "PY3",
    deps = [
    ],
)

py_test(
    name = "read_nist_sphere_test",
    srcs = ["read_nist_sphere_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":read_nist_sphere",
    ],
)

py_binary(
    name = "run_classify_phoneme_on_wav",
    srcs = ["run_classify_phoneme_on_wav.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":classify_phoneme",
        "//extras/python:dsp",
        "//extras/python:frontend",
        "//extras/python:plot",
        "//extras/python/phonetics:phone_util",
        "//extras/python/phonetics:sliding_window",
    ],
)

py_binary(
    name = "run_phone_model_on_wav",
    srcs = ["run_phone_model_on_wav.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":embed_vowel",
        ":hk_util",
        ":phone_model",
        ":phone_util",
        ":sliding_window",
        "//extras/python:dsp",
        "//extras/python:frontend",
        "//extras/python:plot",
    ],
)

py_library(
    name = "sliding_window",
    srcs = ["sliding_window.py"],
    srcs_version = "PY3",
    deps = [
    ],
)

py_test(
    name = "sliding_window_test",
    srcs = ["sliding_window_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sliding_window",
    ],
)

py_library(
    name = "stats",
    srcs = ["stats.py"],
    srcs_version = "PY3",
    deps = [
    ],
)

py_test(
    name = "stats_test",
    srcs = ["stats_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":stats",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":phone_model",
    ],
)
