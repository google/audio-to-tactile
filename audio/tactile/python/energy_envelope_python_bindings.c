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
 * Python bindings for EnergyEnvelope C implementation.
 *
 * These bindings wrap the energy_envelope.c library in audio/tactile as a
 * `energy_envelope` Python module containing an `EnergyEnvelope` class.
 *
 * The interface is as follows.
 *
 * class EnergyEnvelope(object):
 *
 *   def __init__(self,
 *                input_sample_rate_hz=16000.0,
 *                decimation_factor=1,
                  bpf_low_edge_hz,
                  bpf_high_edge_hz,
                  energy_smoother_cutoff_hz,
                  pcen_time_constant_s,
                  pcen_alpha,
                  pcen_beta,
                  pcen_gamma,
                  pcen_delta,
                  output_gain)
 *    """Constructor. [Wraps `EnergyEnvelopeMake()` in the C library.]"""
 *
 *  def Reset():
 *    """Resets to initial state."""
 *
 *  def ProcessSamples(self, input_samples)
 *    """Process samples in a streaming manner.
 *
 *    Args:
 *      input_samples: 1-D numpy array.
 *    Returns:
 *      Array of length `len(input_samples) * decimation_factor`.
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
 */

#include "Python.h"
/* Disallow Numpy 1.7 deprecated symbols. */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "audio/tactile/energy_envelope/energy_envelope.h"
#include "numpy/arrayobject.h"
#include "structmember.h"

typedef struct {
  PyObject_HEAD
  EnergyEnvelope energy_envelope;
  float input_sample_rate_hz;
  float output_sample_rate_hz;
  int decimation_factor;
} EnergyEnvelopeObject;

/* Define `EnergyEnvelope.__init__`. */
static int EnergyEnvelopeObjectInit(EnergyEnvelopeObject* self,
                                    PyObject* args, PyObject* kw) {
  EnergyEnvelopeParams params = kEnergyEnvelopeVowelParams;
  float input_sample_rate_hz = 16000.0f;
  int decimation_factor = 1;
  static const char* keywords[] = {"input_sample_rate_hz",
                                   "decimation_factor",
                                   "bpf_low_edge_hz",
                                   "bpf_high_edge_hz",
                                   "energy_smoother_cutoff_hz",
                                   "pcen_time_constant_s",
                                   "pcen_alpha",
                                   "pcen_beta",
                                   "pcen_gamma",
                                   "pcen_delta",
                                   "output_gain",
                                   NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kw, "|fifffffffff:__init__", (char**)keywords,
          &input_sample_rate_hz,
          &decimation_factor,
          &params.bpf_low_edge_hz,
          &params.bpf_high_edge_hz,
          &params.energy_smoother_cutoff_hz,
          &params.pcen_time_constant_s,
          &params.pcen_alpha,
          &params.pcen_beta,
          &params.pcen_gamma,
          &params.pcen_delta,
          &params.output_gain)) {
    return -1;  /* PyArg_ParseTupleAndKeywords failed. */
  }

  if (!EnergyEnvelopeInit(&self->energy_envelope, &params,
        input_sample_rate_hz, decimation_factor)) {
    PyErr_SetString(PyExc_ValueError, "Error making EnergyEnvelope");
    return -1;
  }
  self->input_sample_rate_hz = input_sample_rate_hz;
  self->decimation_factor = decimation_factor;
  self->output_sample_rate_hz = input_sample_rate_hz / decimation_factor;
  return 0;
}

static void EnergyEnvelopeDealloc(EnergyEnvelopeObject* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

/* Define `EnergyEnvelope.Reset`. */
static PyObject* EnergyEnvelopeObjectReset(EnergyEnvelopeObject* self) {
  EnergyEnvelopeReset(&self->energy_envelope);
  Py_INCREF(Py_None);
  return Py_None;
}

/* Define `EnergyEnvelope.ProcessSamples`. */
static PyObject* EnergyEnvelopeObjectProcessSamples(
    EnergyEnvelopeObject* self, PyObject* args, PyObject* kw) {
  PyObject* samples_arg = NULL;
  static const char* keywords[] = {"samples", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:ProcessSamples",
                                   (char**)keywords, &samples_arg)) {
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
    return NULL;
  }

  const int num_samples = PyArray_SIZE(samples);

  /* Create output numpy array. */
  npy_intp output_dims[1];
  output_dims[0] = num_samples;
  PyArrayObject* output =
      (PyArrayObject*)PyArray_SimpleNew(1, output_dims, NPY_FLOAT);
  if (!output) {  /* PyArray_SimpleNew failed. */
    /* PyArray_SimpleNew already set an error, so clean up and return. */
    Py_XDECREF(samples);
    return NULL;
  }

  /* Process the samples. */
  float* samples_data = (float*)PyArray_DATA(samples);
  float* output_data = (float*)PyArray_DATA(output);
  EnergyEnvelopeProcessSamples(
      &self->energy_envelope, samples_data, num_samples, output_data, 1);

  Py_XDECREF(samples);
  return (PyObject*)output;
}

/* EnergyEnvelope's method functions. */
static PyMethodDef kEnergyEnvelopeMethods[] = {
    {"Reset", (PyCFunction)EnergyEnvelopeObjectReset,
     METH_NOARGS, "Resets to initial state."},
    {"ProcessSamples", (PyCFunction)EnergyEnvelopeObjectProcessSamples,
     METH_VARARGS | METH_KEYWORDS, "Processes samples in a streaming manner."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Define `input_sample_rate_hz`, etc. as read-only members. */
static PyMemberDef kEnergyEnvelopeMembers[] = {
  {"input_sample_rate_hz", T_FLOAT,
   offsetof(EnergyEnvelopeObject, input_sample_rate_hz), READONLY, ""},
  {"output_sample_rate_hz", T_FLOAT,
   offsetof(EnergyEnvelopeObject, output_sample_rate_hz), READONLY, ""},
  {"decimation_factor", T_INT,
   offsetof(EnergyEnvelopeObject, decimation_factor), READONLY, ""},
  {NULL}  /* Sentinel. */
};

/* Define the EnergyEnvelope Python type. */
static PyTypeObject kEnergyEnvelopeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "EnergyEnvelope",                   /* tp_name */
    sizeof(EnergyEnvelopeObject),       /* tp_basicsize */
    0,                                  /* tp_itemsize */
    (destructor)EnergyEnvelopeDealloc,  /* tp_dealloc */
    0,                                  /* tp_print */
    0,                                  /* tp_getattr */
    0,                                  /* tp_setattr */
    0,                                  /* tp_compare */
    0,                                  /* tp_repr */
    0,                                  /* tp_as_number */
    0,                                  /* tp_as_sequence */
    0,                                  /* tp_as_mapping */
    0,                                  /* tp_hash */
    0,                                  /* tp_call */
    0,                                  /* tp_str */
    0,                                  /* tp_getattro */
    0,                                  /* tp_setattro */
    0,                                  /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                 /* tp_flags */
    "EnergyEnvelope object",            /* tp_doc */
    0,                                  /* tp_traverse */
    0,                                  /* tp_clear */
    0,                                  /* tp_richcompare */
    0,                                  /* tp_weaklistoffset */
    0,                                  /* tp_iter */
    0,                                  /* tp_iternext */
    kEnergyEnvelopeMethods,             /* tp_methods */
    kEnergyEnvelopeMembers,             /* tp_members */
    0,                                  /* tp_getset */
    0,                                  /* tp_base */
    0,                                  /* tp_dict */
    0,                                  /* tp_descr_get */
    0,                                  /* tp_descr_set */
    0,                                  /* tp_dictoffset */
    (initproc)EnergyEnvelopeObjectInit, /* tp_init */
};

static void InitModule(PyObject* m) {
  kEnergyEnvelopeType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&kEnergyEnvelopeType) >= 0) {
    Py_INCREF(&kEnergyEnvelopeType);
    PyModule_AddObject(m, "EnergyEnvelope", (PyObject*)&kEnergyEnvelopeType);
  }
}

/* Module methods. */
static PyMethodDef kModuleMethods[] = {
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Python 3 does module initialization differently. */
#if PY_MAJOR_VERSION >= 3
/* Module definition. */
static struct PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "energy_envelope",  /* m_name */
    NULL,               /* m_doc */
    (Py_ssize_t)-1,     /* m_size */
    kModuleMethods,     /* m_methods */
    NULL,               /* m_reload */
    NULL,               /* m_traverse */
    NULL,               /* m_clear */
    NULL,               /* m_free */
};

PyMODINIT_FUNC PyInit_energy_envelope() {
  import_array();
  PyObject* m = PyModule_Create(&kModule);
  InitModule(m);
  return m;
}
#else
PyMODINIT_FUNC initenergy_envelope() {
  import_array();
  PyObject* m = Py_InitModule("energy_envelope", kModuleMethods);
  InitModule(m);
}
#endif

