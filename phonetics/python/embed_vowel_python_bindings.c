/* Copyright 2019 Google LLC
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
 * Python bindings for vowel embedding C implementation.
 *
 * These bindings wrap the embed_vowel.c library in
 * phonetics as a `embed_vowel` Python module containing a
 * `EmbedVowel` Python function.
 *
 * The interface is as follows. See also embed_vowel_test.py for use example.
 *
 * def embed_vowel(frame):
 *   """Vowel embedding inference. [Wraps `EmbedVowel()` in the C library.]
 *
 *   Args:
 *     frame: 2D array of shape [num_frames, NUM_CHANNELS].
 *   Returns:
 *     2D array of shape [num_frames, 2], where the ith row is the predicted
 *     vowel space coordinate of the ith frame.
 *   """
 *
 * def embed_vowel_scores(frame):
 *   """Computes scores for each target.
 *
 *   Args:
 *     frame: 2D array of shape [num_frames, NUM_CHANNELS].
 *   Returns:
 *     2D array of shape [num_frames, NUM_TARGETS], where the ith row contains
 *     the predicted vowel scores for the ith frame.
 *   """
 *
 * The expected number of CARL channels is exposed as a module constant
 * `NUM_CHANNELS` and the embedding targets as a dictionary `TARGETS`.
 * `TARGETS[phone]` is the 2D target coordinate for `phone`.
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
 * https://docs.scipy.org/doc/numpy/reference/c-api.array.html
 */

#include <string.h>

#include "Python.h"
/* Disallow Numpy 1.7 deprecated symbols. */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "phonetics/embed_vowel.h"
#include "numpy/arrayobject.h"

#define NUM_CHANNELS kEmbedVowelNumChannels
#define NUM_TARGETS kEmbedVowelNumTargets

/* Convert input arg to numpy array with contiguous float32 data. The caller
 * must call Py_XDECREF on the returned object to release memory.
 */
static PyArrayObject* GetInputFrameArray(PyObject* frame_arg) {
  PyArrayObject* frame = (PyArrayObject*)PyArray_FromAny(
      frame_arg, PyArray_DescrFromType(NPY_FLOAT), 0, 0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_FORCECAST |
          NPY_ARRAY_DEFAULT,
      NULL);

  if (!frame) {  /* PyArray_DescrFromType failed. */
    /* PyArray_DescrFromType already set an error, so just need to return. */
    return NULL;
  } else if (PyArray_NDIM(frame) != 2 ||
             PyArray_DIM(frame, 1) != NUM_CHANNELS) {
    Py_XDECREF(frame);
    PyErr_SetString(PyExc_ValueError,
        "input must be a 2D array with NUM_CHANNELS columns");
    return NULL;
  }
  return frame;
}

static PyArrayObject* MakeOutputArray(int num_rows, int num_cols) {
  npy_intp output_dims[2];
  output_dims[0] = num_rows;
  output_dims[1] = num_cols;
  return (PyArrayObject*)PyArray_SimpleNew(2, output_dims, NPY_FLOAT);
}

/* Define `embed_vowel()`. */
static PyObject* EmbedVowelPython(
    PyObject* dummy, PyObject* args, PyObject* kw) {
  PyObject* frame_arg;
  static const char* keywords[] = {"frame", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:embed_vowel", (char**)keywords,
                                   &frame_arg)) {
    return NULL;  /* PyArg_ParseTupleAndKeywords failed. */
  }

  PyArrayObject* frame;
  if (!(frame = GetInputFrameArray(frame_arg))) {
    return NULL;
  }
  const int num_frames = PyArray_DIM(frame, 0);
  PyArrayObject* output;
  if (!(output = MakeOutputArray(num_frames, 2))) {
    Py_XDECREF(frame);
    return NULL;
  }

  /* Perform the vowel mapping. */
  const float* frame_data = (const float*)PyArray_DATA(frame);
  float* output_data = (float*)PyArray_DATA(output);
  int i;
  for (i = 0; i < num_frames; ++i) {
    EmbedVowel(frame_data, output_data);
    frame_data += kEmbedVowelNumChannels;
    output_data += 2;
  }

  Py_XDECREF(frame);
  return (PyObject*)output;
}

static float ComputeScore(const EmbedVowelTarget* target, const float* coord) {
  const float diff_x = target->coord[0] - coord[0];
  const float diff_y = target->coord[1] - coord[1];
  const float distance = sqrt(diff_x * diff_x + diff_y * diff_y);
  return exp(-4.0f * distance);
}

/* Define `embed_vowel_scores()`. */
static PyObject* EmbedVowelScoresPython(
    PyObject* dummy, PyObject* args, PyObject* kw) {
  PyObject* frame_arg;
  static const char* keywords[] = {"frame", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args, kw, "O:embed_vowel_scores", (char**)keywords, &frame_arg)) {
    return NULL;  /* PyArg_ParseTupleAndKeywords failed. */
  }

  PyArrayObject* frame;
  if (!(frame = GetInputFrameArray(frame_arg))) {
    return NULL;
  }
  const int num_frames = PyArray_DIM(frame, 0);
  PyArrayObject* output;
  if (!(output = MakeOutputArray(num_frames, kEmbedVowelNumTargets))) {
    Py_XDECREF(frame);
    return NULL;
  }

  /* Call EmbedVowelScores. */
  const float* frame_data = (const float*)PyArray_DATA(frame);
  float* output_data = (float*)PyArray_DATA(output);
  int i;
  for (i = 0; i < num_frames; ++i) {
    float coord[2];
    EmbedVowel(frame_data, coord);

    int j;
    for (j = 0; j < kEmbedVowelNumTargets; ++j) {
      output_data[j] = ComputeScore(&kEmbedVowelTargets[j], coord);
    }

    frame_data += kEmbedVowelNumChannels;
    output_data += kEmbedVowelNumTargets;
  }

  Py_XDECREF(frame);
  return (PyObject*)output;
}

/* Module methods. */
static PyMethodDef kModuleMethods[] = {
    {"embed_vowel", (PyCFunction)EmbedVowelPython, METH_VARARGS | METH_KEYWORDS,
     "Performs vowel embedding inference."},
    {"embed_vowel_scores", (PyCFunction)EmbedVowelScoresPython,
      METH_VARARGS | METH_KEYWORDS, "Computes scores for each target."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static void InitModule(PyObject* m) {
  /* Make `NUM_CHANNELS` module int constant. */
  PyModule_AddIntMacro(m, NUM_CHANNELS);
  /* Make `NUM_TARGETS` module int constant. */
  PyModule_AddIntMacro(m, NUM_TARGETS);

  /* Make `TARGET_NAMES` list. */
  PyObject* target_names = PyList_New(NUM_TARGETS);
  if (!target_names) { return; }
  int i;
  for (i = 0; i < kEmbedVowelNumTargets; ++i) {
    PyObject* name = PyUnicode_FromString(kEmbedVowelTargets[i].name);
    if (PyList_SetItem(target_names, i, name)) {
      Py_XDECREF(name);
      Py_XDECREF(target_names);
      return;
    }
  }
  PyModule_AddObject(m, "TARGET_NAMES", target_names);

  npy_intp dims[2];
  dims[0] = NUM_TARGETS;
  dims[1] = 2;
  PyObject* target_coords = PyArray_SimpleNew(2, dims, NPY_FLOAT);
  if (!target_coords) {
    return;
  }
  float* dest = PyArray_DATA((PyArrayObject*)target_coords);
  for (i = 0; i < kEmbedVowelNumTargets; ++i, dest += 2) {
    memcpy(dest, kEmbedVowelTargets[i].coord, sizeof(float) * 2);
  }
  PyModule_AddObject(m, "TARGET_COORDS", target_coords);
}

/* Module definition. */
static struct PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "embed_vowel",   /* m_name */
    NULL,            /* m_doc */
    (Py_ssize_t)-1,  /* m_size */
    kModuleMethods,  /* m_methods */
    NULL,            /* m_reload */
    NULL,            /* m_traverse */
    NULL,            /* m_clear */
    NULL,            /* m_free */
};

PyMODINIT_FUNC PyInit_embed_vowel() {
  import_array();
  PyObject* m = PyModule_Create(&kModule);
  InitModule(m);
  return m;
}
