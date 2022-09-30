# Copyright 2019, 2021 Google LLC
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

"""Bazel support for Python extensions and C.

Defines py_extension for compiling a Python extension module from C/C++ code.

Defines c_binary, c_library, c_test rules for targets written in C. They are
like their cc_* counterparts, but compile with C89 standard compatibility.
"""

PYTHON_INCLUDE_DIRS = [
    "/usr/include/python3.9",
    "/usr/include/x86_64-linux-gnu/python3.9",
]

def py_extension(name = None, srcs = None, data = None, visibility = None, deps = None):
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

CONDITION_WINDOWS = "@bazel_tools//src/conditions:windows"

WARNING_OPTS = [
    # Warn about mixed signed-unsigned integer comparisons.
    "-Wsign-compare",
    # Warn about `()` declarations. In C, but not C++, a function taking no args
    # should be declared as `void foo(void);`, not `void foo();`.
    # See e.g. https://stackoverflow.com/a/13950800/13223986.
    "-Wstrict-prototypes",
    # Suppress "unused function" warnings on `static` functions in .h files.
    "-Wno-unused-function",
] + select({
    CONDITION_WINDOWS: [],
    # Warn about needlessly set variables. Not supported on Windows.
    "//conditions:default": ["-Wunused-but-set-variable"],
})

# Build with C89 standard compatibility.
DEFAULT_C_OPTS = WARNING_OPTS + ["-std=c89"]

def c_binary(name = None, **kwargs):
    """cc_binary with DEFAULT_COPTS."""
    kwargs.update({"copts": DEFAULT_C_OPTS + kwargs.get("copts", [])})
    return native.cc_binary(name = name, **kwargs)

def c_library(name = None, **kwargs):
    """cc_library with DEFAULT_C_OPTS, and hdrs is used as textual_hrds."""
    kwargs.update({"copts": DEFAULT_C_OPTS + kwargs.get("copts", [])})

    # Use "hdrs" as "textual_hdrs". All code that cannot be standalone-compiled
    # as C++ must be listed in textual_hdrs.
    kwargs.setdefault("textual_hdrs", kwargs.pop("hdrs", None))
    return native.cc_library(name = name, **kwargs)

def c_test(name = None, **kwargs):
    """cc_test with DEFAULT_COPTS."""
    kwargs.update({"copts": DEFAULT_C_OPTS + kwargs.get("copts", [])})
    return native.cc_test(name = name, **kwargs)
