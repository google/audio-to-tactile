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
# Code to wrap a Jupyter GUI around simple tactile experiments.

licenses(["notice"])

py_test(
    name = "levitt_experiment_test",
    srcs = ["levitt_experiment_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":levitt_experiment",
    ],
)

py_library(
    name = "levitt_experiment",
    srcs = ["levitt_experiment.py"],
    srcs_version = "PY3",
    deps = [
    ],
)
