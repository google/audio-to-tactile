/* Copyright 2020-2022 Google LLC
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
 * Python bindings for Enveloper C implementation.
 *
 * These bindings wrap the energy_envelope.c library in tactile as an
 * `enveloper` Python module containing an `Enveloper` class.
 *
 * The interface is as follows.
 *
 * class Enveloper(object):
 *
 *   def __init__(self,
 *                input_sample_rate_hz,
 *                decimation_factor=1,
 *                bpf_low_edge_hz=(80.0, 500.0, 2500.0, 4000.0),
 *                bpf_high_edge_hz=(500.0, 3500.0, 3500.0, 6000.0),
 *                energy_cutoff_hz=500.0,
 *                noise_db_s=2.0,
 *                agc_strength=0.7,
 *                compressor_exponent=0.25,
 *                output_gain=(1.0, 1.0, 1.0, 1.0))
 *    """Constructor. [Wraps `EnveloperInit()` in the C library.]"""
 *
 *  def reset():
 *    """Resets to initial state."""
 *
 *  def process_samples(self, input_samples, debug_out=None)
 *    """Process samples in a streaming manner.
 *
 *    Args:
 *      input_samples: 1-D numpy array.
 *      debug_out: (Optional) A dict, which if passed, is filled with debug
 *        signals of the Enveloper's internal state.
 *    Returns:
 *      Array of length `len(input_samples) / decimation_factor`.
 *    """
 *
 *  @property
 *  def input_sample_rate_hz(self)
 *    """Sample rate of the input audio."""
 *
 *  @property
 *  def output_sample_rate_hz(self)
 *    """Sample rate of the output tactile signals."""
 *
 *  @property
 *  def decimation_factor(self)
 *    """Decimation factor between input and output."""
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

#define PY_SSIZE_T_CLEAN
#include "Python.h"
/* Disallow Numpy 1.7 deprecated symbols. */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "src/tactile/enveloper.h"
#include "numpy/arrayobject.h"
#include "structmember.h"

typedef struct {
  PyObject_HEAD
  Enveloper enveloper;
  float input_sample_rate_hz;
  float output_sample_rate_hz;
  int decimation_factor;
} EnveloperObject;

/* Define `Enveloper.__init__`. */
static int EnveloperObjectInit(EnveloperObject* self,
                               PyObject* args, PyObject* kw) {
  EnveloperParams params = kDefaultEnveloperParams;
  float input_sample_rate_hz = 16000.0f;
  int decimation_factor = 1;
  static const char* keywords[] = {"input_sample_rate_hz",
                                   "decimation_factor",
                                   "bpf_low_edge_hz",
                                   "bpf_high_edge_hz",
                                   "energy_cutoff_hz",
                                   "noise_db_s",
                                   "agc_strength",
                                   "compressor_exponent",
                                   "output_gain",
                                   NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kw,
          "f|i(ffff)(ffff)ffff(ffff):__init__", (char**)keywords,
          &input_sample_rate_hz,
          &decimation_factor,
          &params.channel_params[0].bpf_low_edge_hz,
          &params.channel_params[1].bpf_low_edge_hz,
          &params.channel_params[2].bpf_low_edge_hz,
          &params.channel_params[3].bpf_low_edge_hz,
          &params.channel_params[0].bpf_high_edge_hz,
          &params.channel_params[1].bpf_high_edge_hz,
          &params.channel_params[2].bpf_high_edge_hz,
          &params.channel_params[3].bpf_high_edge_hz,
          &params.energy_cutoff_hz,
          &params.noise_db_s,
          &params.agc_strength,
          &params.compressor_exponent,
          &params.channel_params[0].output_gain,
          &params.channel_params[1].output_gain,
          &params.channel_params[2].output_gain,
          &params.channel_params[3].output_gain)) {
    return -1;  /* PyArg_ParseTupleAndKeywords failed. */
  }

  if (!EnveloperInit(&self->enveloper, &params,
        input_sample_rate_hz, decimation_factor)) {
    PyErr_SetString(PyExc_ValueError, "Error making Enveloper");
    return -1;
  }
  self->input_sample_rate_hz = input_sample_rate_hz;
  self->decimation_factor = decimation_factor;
  self->output_sample_rate_hz = input_sample_rate_hz / decimation_factor;
  return 0;
}

static void EnveloperDealloc(EnveloperObject* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

/* Define `Enveloper.reset`. */
static PyObject* EnveloperObjectReset(EnveloperObject* self) {
  EnveloperReset(&self->enveloper);
  Py_INCREF(Py_None);
  return Py_None;
}

/* Creates 1D float array in dict. Returns data pointer, or NULL on failure. */
static float* CreateArrayInDict(PyObject* dict, const char* key,
                                int num_frames) {
  npy_intp dims[2];
  dims[0] = num_frames;
  dims[1] = kEnveloperNumChannels;
  PyObject* array = PyArray_SimpleNew(2, dims, NPY_FLOAT);
  if (!array) { return NULL; }
  float* data = (float*)PyArray_DATA((PyArrayObject*)array);
  const int success = (PyDict_SetItemString(dict, key, array) == 0);
  /* PyDict_SetItemString() does *not* steal a reference to array. */
  Py_DECREF(array);
  return success ? data : NULL;
}

/* Define `Enveloper.process_samples`. */
static PyObject* EnveloperObjectProcessSamples(
    EnveloperObject* self, PyObject* args, PyObject* kw) {
  PyObject* samples_arg = NULL;
  PyObject* debug_out = NULL;
  static const char* keywords[] = {"samples", "debug_out", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|O!:process_samples",
                                   (char**)keywords, &samples_arg,
                                   &PyDict_Type, &debug_out)) {
    return NULL;  /* PyArg_ParseTupleAndKeywords failed. */
  }

  /* Convert input samples to numpy array with contiguous float32 data. */
  PyArrayObject* samples = (PyArrayObject*)PyArray_FromAny(
      samples_arg, PyArray_DescrFromType(NPY_FLOAT), 0, 0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_FORCECAST |
          NPY_ARRAY_DEFAULT,
      NULL);

  if (!samples) {  /* PyArray_DescrFromType failed. */
    /* PyArray_DescrFromType already set an error, so just need to return. */
    return NULL;
  } else if (PyArray_NDIM(samples) != 1) {
    PyErr_SetString(PyExc_ValueError, "expected 1-D array");
    Py_DECREF(samples);
    return NULL;
  }

  const int num_samples = PyArray_SIZE(samples);
  const float* samples_data = (const float*)PyArray_DATA(samples);
  const int output_frames = num_samples / self->decimation_factor;

  npy_intp output_dims[2];
  output_dims[0] = output_frames;
  output_dims[1] = kEnveloperNumChannels;
  PyArrayObject* output_array =
      (PyArrayObject*)PyArray_SimpleNew(2, output_dims, NPY_FLOAT);
  if (!output_array) {
    Py_DECREF(samples);
    return NULL;
  }
  float* output = (float*)PyArray_DATA(output_array);

  if (!debug_out) {
    /* Process the samples. */
    EnveloperProcessSamples(
        &self->enveloper, samples_data, num_samples, output);
  } else {
    /* If a `debug_out` dict was passed, clear it and put arrays in it for
     * holding energy envelope debug signals.
     */
    PyDict_Clear(debug_out);
    float* smoothed_energy = NULL;
    float* noise = NULL;
    if (!(smoothed_energy = CreateArrayInDict(
          debug_out, "smoothed_energy", output_frames)) ||
        !(noise = CreateArrayInDict(
          debug_out, "noise", output_frames))) {
      Py_DECREF(samples);
      Py_DECREF(output_array);
      PyDict_Clear(debug_out);
      return NULL;
    }

    int i;
    for (i = 0; i < output_frames; ++i) {
      /* Process the samples. In order to save debug signals, we pass
       * decimation_factor input samples so that one output sample is produced
       * at a time.
       */
      EnveloperProcessSamples(
          &self->enveloper, samples_data, self->decimation_factor,
          output);

      int c;
      for (c = 0; c < kEnveloperNumChannels; ++c) {
        smoothed_energy[c] = self->enveloper.channels[c].smoothed_energy;
        noise[c] = self->enveloper.channels[c].noise;
      }

      samples_data += self->decimation_factor;
      output += kEnveloperNumChannels;
      smoothed_energy += kEnveloperNumChannels;
      noise += kEnveloperNumChannels;
    }
  }


  Py_DECREF(samples);
  return (PyObject*)output_array;
}

/* Enveloper's method functions. */
static PyMethodDef kEnveloperMethods[] = {
    {"reset", (PyCFunction)EnveloperObjectReset,
     METH_NOARGS, "Resets to initial state."},
    {"process_samples", (PyCFunction)EnveloperObjectProcessSamples,
     METH_VARARGS | METH_KEYWORDS, "Processes samples in a streaming manner."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Define `input_sample_rate_hz`, etc. as read-only members. */
static PyMemberDef kEnveloperMembers[] = {
  {"input_sample_rate_hz", T_FLOAT,
   offsetof(EnveloperObject, input_sample_rate_hz), READONLY, ""},
  {"output_sample_rate_hz", T_FLOAT,
   offsetof(EnveloperObject, output_sample_rate_hz), READONLY, ""},
  {"decimation_factor", T_INT,
   offsetof(EnveloperObject, decimation_factor), READONLY, ""},
  {NULL}  /* Sentinel. */
};

/* Define the Enveloper Python type. */
static PyTypeObject kEnveloperType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "Enveloper",                   /* tp_name */
    sizeof(EnveloperObject),       /* tp_basicsize */
    0,                             /* tp_itemsize */
    (destructor)EnveloperDealloc,  /* tp_dealloc */
    0,                             /* tp_print */
    0,                             /* tp_getattr */
    0,                             /* tp_setattr */
    0,                             /* tp_compare */
    0,                             /* tp_repr */
    0,                             /* tp_as_number */
    0,                             /* tp_as_sequence */
    0,                             /* tp_as_mapping */
    0,                             /* tp_hash */
    0,                             /* tp_call */
    0,                             /* tp_str */
    0,                             /* tp_getattro */
    0,                             /* tp_setattro */
    0,                             /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,            /* tp_flags */
    "Enveloper object",            /* tp_doc */
    0,                             /* tp_traverse */
    0,                             /* tp_clear */
    0,                             /* tp_richcompare */
    0,                             /* tp_weaklistoffset */
    0,                             /* tp_iter */
    0,                             /* tp_iternext */
    kEnveloperMethods,             /* tp_methods */
    kEnveloperMembers,             /* tp_members */
    0,                             /* tp_getset */
    0,                             /* tp_base */
    0,                             /* tp_dict */
    0,                             /* tp_descr_get */
    0,                             /* tp_descr_set */
    0,                             /* tp_dictoffset */
    (initproc)EnveloperObjectInit, /* tp_init */
};

/* Module methods. */
static PyMethodDef kModuleMethods[] = {
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Module definition. */
static struct PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "enveloper",    /* m_name */
    NULL,           /* m_doc */
    (Py_ssize_t)-1, /* m_size */
    kModuleMethods, /* m_methods */
    NULL,           /* m_reload */
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC PyInit_enveloper(void) {
  import_array();
  PyObject* m = PyModule_Create(&kModule);
  PyModule_AddIntConstant(m, "NUM_CHANNELS", kEnveloperNumChannels);
  kEnveloperType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&kEnveloperType) >= 0) {
    Py_INCREF(&kEnveloperType);
    PyModule_AddObject(m, "Enveloper", (PyObject*)&kEnveloperType);
  }
  return m;
}
