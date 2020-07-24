# Copyright 2020 Google LLC
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
# JAX/Haiku utilities.

package(default_visibility = ["//visibility:public"])

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

