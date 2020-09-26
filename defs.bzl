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

PYTHON_INCLUDE_DIRS = [
    "/usr/include/python3.8",
    "/usr/include/x86_64-linux-gnu/python3.8",
]

WARNING_OPTS = [
    "-Wno-unused-function",
    "-Wno-unused-parameter",
]

DEFAULT_C_OPTS = WARNING_OPTS + ["-std=c89"]


def c_binary(**kwargs):
  kwargs.update({'copts': DEFAULT_C_OPTS + kwargs.get('copts', [])})
  return native.cc_binary(**kwargs)


def c_library(**kwargs):
  kwargs.update({'copts': DEFAULT_C_OPTS + kwargs.get('copts', [])})
  kwargs.setdefault('textual_hdrs', kwargs.pop('hdrs', None))
  return native.cc_library(**kwargs)


def c_test(**kwargs):
  kwargs.update({'copts': DEFAULT_C_OPTS + kwargs.get('copts', [])})
  return native.cc_test(**kwargs)


def py_extension(name=None, srcs=None, data=None, visibility=None, deps=None):

    binary_name = name + ".so"

    native.cc_binary(
        name = binary_name,
        linkshared = True,
        linkstatic = True,
        visibility = ["//visibility:private"],
        srcs = srcs,
        data = data,
        deps = deps,
        copts = WARNING_OPTS + ["-I" + d for d in PYTHON_INCLUDE_DIRS],
        linkopts = ["-fPIC"],
    )

    native.py_library(
        name = name,
        data = [binary_name],
        visibility = visibility,
    )
