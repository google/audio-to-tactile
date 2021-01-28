# Copyright 2020-2021 Google LLC
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
# Audio-to-tactile: Tactile interfaces for hearing accessibility.

load(":defs.bzl", "c_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

c_library(
    name = "dsp",
    srcs = glob(["src/dsp/*.c"]),
    hdrs = glob(["src/dsp/*.h"]),
    # Add "src" as an include directory for "dsp" and all targets that depend
    # on "dsp". This lets us write e.g. #include "dsp/fast_fun.h" instead of
    # "third_party/audio_to_tactile/src/dsp/fast_fun.h".
    includes = ["src"],
)

c_library(
    name = "frontend",
    srcs = glob(["src/frontend/*.c"]),
    hdrs = glob(["src/frontend/*.h"]),
    includes = ["src"],
    deps = [
        ":dsp",
    ],
)

c_library(
    name = "mux",
    srcs = glob(["src/mux/*.c"]),
    hdrs = glob(["src/mux/*.h"]),
    includes = ["src"],
    deps = [
        ":dsp",
    ],
)

c_library(
    name = "phonetics",
    srcs = glob(["src/phonetics/*.c"]),
    hdrs = glob(["src/phonetics/*.h"]),
    includes = ["src"],
    deps = [
        ":dsp",
    ],
)

c_library(
    name = "tactile",
    srcs = glob(["src/tactile/*.c"]),
    hdrs = glob(["src/tactile/*.h"]),
    includes = ["src"],
    deps = [
        ":dsp",
        ":frontend",
        ":phonetics",
    ],
)

filegroup(
    name = "repo_files",
    srcs = glob(
        [
            "*",
            "examples/**",
            "extras/**",
            "src/**",
        ],
    ),
)
