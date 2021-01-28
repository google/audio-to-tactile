/* Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * Python bindings for fast_fun C library.
 *
 * NOTE: Using a tool like CLIF is generally a better idea than writing bindings
 * manually. We do it here anyway since as open sourced code it matters that it
 * is easy for others to build. Also, we use numpy, which CLIF does not natively
 * support, and it is easier to use numpy's (excellent) C API directly than it
 * is to write CLIF custom type glue code.
 *
 * For general background on C/C++ Python extensions, see the intro guide
 * https://docs.python.org/3/extending/extending.html
 * Separately, numpy has a C API to interact with its types, see
 * https://numpy.org/doc/stable/reference/c-api/array.html
 */

#include "Python.h"
/* Disallow Numpy 1.7 deprecated symbols. */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "src/dsp/fast_fun.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

/* These Array* functions used below to create numpy "universal functions";
 * see [https://numpy.org/doc/stable/user/c-info.ufunc-tutorial.html].
 */

static void ArrayFastLog2(char** args, npy_intp* dimensions,
                          npy_intp* steps, void* unused_data) {
  char* in = args[0];
  char* out = args[1];
  const npy_intp n = dimensions[0];
  const npy_intp in_step = steps[0];
  const npy_intp out_step = steps[1];
  npy_intp i;

  for (i = 0; i < n; ++i) {
    *((float*)out) = FastLog2(*(float*)in);
    in += in_step;
    out += out_step;
  }
}

static void ArrayFastExp2(char** args, npy_intp* dimensions,
                          npy_intp* steps, void* unused_data) {
  char* in = args[0];
  char* out = args[1];
  const npy_intp n = dimensions[0];
  const npy_intp in_step = steps[0];
  const npy_intp out_step = steps[1];
  npy_intp i;

  for (i = 0; i < n; ++i) {
    *((float*)out) = FastExp2(*(float*)in);
    in += in_step;
    out += out_step;
  }
}

static void ArrayFastTanh(char** args, npy_intp* dimensions,
                          npy_intp* steps, void* unused_data) {
  char* in = args[0];
  char* out = args[1];
  const npy_intp n = dimensions[0];
  const npy_intp in_step = steps[0];
  const npy_intp out_step = steps[1];
  npy_intp i;

  for (i = 0; i < n; ++i) {
    *((float*)out) = FastTanh(*(float*)in);
    in += in_step;
    out += out_step;
  }
}

static PyUFuncGenericFunction kFastLog2Funs[1] = {&ArrayFastLog2};
static PyUFuncGenericFunction kFastExp2Funs[1] = {&ArrayFastExp2};
static PyUFuncGenericFunction kFastTanhFuns[1] = {&ArrayFastTanh};

static PyMethodDef kModuleMethods[] = {
  {NULL, NULL, 0, NULL}
};

static int AddUFuncToDict(PyObject* dict,
                          const char* name,
                          PyUFuncGenericFunction* funs) {
  if (!dict) { return 0; }

  /* Create universal function mapping a float32 input to a float32 output. */
  static void* kUFuncNoData[1] = {NULL};
  static char kUFuncFloatToFloat[2] = {NPY_FLOAT, NPY_FLOAT};
  PyObject* ufunc =
      PyUFunc_FromFuncAndData(funs, kUFuncNoData, kUFuncFloatToFloat, 1, 1, 1,
                              PyUFunc_None, name, name, 0);
  if (!ufunc) { return 0; }

  /* Add the ufunc to `dict`. */
  const int success = (PyDict_SetItemString(dict, name, ufunc) == 0);
  Py_DECREF(ufunc);
  return success;
}

/* Module definition. */
static struct PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "fast_fun_python_bindings",  /* m_name */
    NULL,                        /* m_doc */
    (Py_ssize_t)-1,              /* m_size */
    kModuleMethods,              /* m_methods */
    NULL,                        /* m_reload */
    NULL,                        /* m_traverse */
    NULL,                        /* m_clear */
    NULL,                        /* m_free */
};

PyMODINIT_FUNC PyInit_fast_fun_python_bindings() {
  import_array();
  import_umath();

  PyObject* m = PyModule_Create(&kModule);
  if (!m) { return NULL; }

  PyObject* dict = PyModule_GetDict(m);
  if (!AddUFuncToDict(dict, "fast_log2_impl", kFastLog2Funs) ||
      !AddUFuncToDict(dict, "fast_exp2_impl", kFastExp2Funs) ||
      !AddUFuncToDict(dict, "fast_tanh_impl", kFastTanhFuns)) {
    return NULL;
  }

  return m;
}
