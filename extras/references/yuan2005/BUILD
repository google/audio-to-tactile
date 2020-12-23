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
# Envelope onset asynchrony (EOA) tactile feature of Yuan et al.

load("//:defs.bzl", "c_library", "c_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

c_library(
    name = "yuan2005",
    srcs = ["yuan2005.c"],
    hdrs = ["yuan2005.h"],
    deps = [
        "//:dsp",
    ],
)

c_test(
    name = "yuan2005_test",
    srcs = ["yuan2005_test.c"],
    deps = [
        ":yuan2005",
        "//:dsp",
    ],
)
