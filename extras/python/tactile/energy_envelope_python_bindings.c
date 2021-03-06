/* Copyright 2020-2021 Google LLC
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
 * These bindings wrap the energy_envelope.c library in tactile as a
 * `energy_envelope` Python module containing an `EnergyEnvelope` class.
 *
 * The interface is as follows.
 *
 * class EnergyEnvelope(object):
 *
 *   def __init__(self,
 *                input_sample_rate_hz,
 *                decimation_factor=1,
 *                bpf_low_edge_hz=500.0,
 *                bpf_high_edge_hz=3500.0,
 *                energy_cutoff_hz=500.0,
 *                energy_tau_s=0.01,
 *                noise_tau_s=0.4,
 *                agc_strength=0.7,
 *                denoise_thresh_factor=8.0,
 *                gain_tau_attack_s=0.002,
 *                gain_tau_release_s=0.15,
 *                compressor_exponent=0.25,
 *                compressor_delta=0.01,
 *                output_gain=1.0)
 *    """Constructor. [Wraps `EnergyEnvelopeMake()` in the C library.]"""
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
 *        signals of energy envelope's internal state.
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

#include "Python.h"
/* Disallow Numpy 1.7 deprecated symbols. */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "src/tactile/energy_envelope.h"
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
                                   "energy_cutoff_hz",
                                   "energy_tau_s",
                                   "noise_tau_s",
                                   "agc_strength",
                                   "denoise_thresh_factor",
                                   "gain_tau_attack_s",
                                   "gain_tau_release_s",
                                   "compressor_exponent",
                                   "compressor_delta",
                                   "output_gain",
                                   NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kw, "f|iffffffffffff:__init__", (char**)keywords,
          &input_sample_rate_hz,
          &decimation_factor,
          &params.bpf_low_edge_hz,
          &params.bpf_high_edge_hz,
          &params.energy_cutoff_hz,
          &params.energy_tau_s,
          &params.noise_tau_s,
          &params.agc_strength,
          &params.denoise_thresh_factor,
          &params.gain_tau_attack_s,
          &params.gain_tau_release_s,
          &params.compressor_exponent,
          &params.compressor_delta,
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

/* Define `EnergyEnvelope.reset`. */
static PyObject* EnergyEnvelopeObjectReset(EnergyEnvelopeObject* self) {
  EnergyEnvelopeReset(&self->energy_envelope);
  Py_INCREF(Py_None);
  return Py_None;
}

/* Creates 1D float array in dict. Returns data pointer, or NULL on failure. */
static float* CreateArrayInDict(PyObject* dict, const char* key,
                                npy_intp size) {
  PyObject* array = PyArray_SimpleNew(1, &size, NPY_FLOAT);
  if (!array) { return NULL; }
  float* data = (float*)PyArray_DATA((PyArrayObject*)array);
  const int success = (PyDict_SetItemString(dict, key, array) == 0);
  /* PyDict_SetItemString() does *not* steal a reference to array. */
  Py_DECREF(array);
  return success ? data : NULL;
}

/* Define `EnergyEnvelope.process_samples`. */
static PyObject* EnergyEnvelopeObjectProcessSamples(
    EnergyEnvelopeObject* self, PyObject* args, PyObject* kw) {
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
  npy_intp output_size = num_samples / self->decimation_factor;

  PyArrayObject* output_array =
      (PyArrayObject*)PyArray_SimpleNew(1, &output_size, NPY_FLOAT);
  if (!output_array) {
    Py_DECREF(samples);
    return NULL;
  }
  float* output = (float*)PyArray_DATA(output_array);

  if (!debug_out) {
    /* Process the samples. */
    EnergyEnvelopeProcessSamples(
        &self->energy_envelope, samples_data, num_samples, output, 1);
  } else {
    /* If a `debug_out` dict was passed, clear it and put arrays in it for
     * holding energy envelope debug signals.
     */
    PyDict_Clear(debug_out);
    float* smoothed_energy = NULL;
    float* log2_noise = NULL;
    float* smoothed_gain = NULL;
    if (!(smoothed_energy = CreateArrayInDict(
            debug_out, "smoothed_energy", output_size)) ||
        !(log2_noise = CreateArrayInDict(
            debug_out, "log2_noise", output_size)) ||
        !(smoothed_gain = CreateArrayInDict(
            debug_out, "smoothed_gain", output_size))) {
      Py_DECREF(samples);
      Py_DECREF(output_array);
      PyDict_Clear(debug_out);
      return NULL;
    }

    int i;
    for (i = 0; i < output_size; ++i) {
      /* Process the samples. In order to save debug signals, we pass
       * decimation_factor input samples so that one output sample is produced
       * at a time.
       */
      EnergyEnvelopeProcessSamples(
          &self->energy_envelope, samples_data, self->decimation_factor,
          output + i, 1);
      samples_data += self->decimation_factor;

      smoothed_energy[i] = self->energy_envelope.smoothed_energy;
      log2_noise[i] = self->energy_envelope.log2_noise;
      smoothed_gain[i] = self->energy_envelope.smoothed_gain;
    }
  }


  Py_DECREF(samples);
  return (PyObject*)output_array;
}

/* EnergyEnvelope's method functions. */
static PyMethodDef kEnergyEnvelopeMethods[] = {
    {"reset", (PyCFunction)EnergyEnvelopeObjectReset,
     METH_NOARGS, "Resets to initial state."},
    {"process_samples", (PyCFunction)EnergyEnvelopeObjectProcessSamples,
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

/* Module methods. */
static PyMethodDef kModuleMethods[] = {
    {NULL, NULL, 0, NULL} /* Sentinel */
};

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

/* Sets a float value in `dict`. Returns 1 on success, 0 on failure. */
static int SetFloatInDict(PyObject* dict, const char* key, float value) {
  PyObject* value_obj = PyFloat_FromDouble(value);
  if (!value_obj) { return 0; }
  const int success = (PyDict_SetItemString(dict, key, value_obj) == 0);
  /* PyDict_SetItemString() does *not* steal a reference to value_obj. */
  Py_DECREF(value_obj);
  return success;
}

static void AddParamsDict(PyObject* m, const char* name,
                          const EnergyEnvelopeParams* params) {
  PyObject* dict = PyDict_New();
  if (!dict ||
      !SetFloatInDict(dict, "bpf_low_edge_hz", params->bpf_low_edge_hz) ||
      !SetFloatInDict(dict, "bpf_high_edge_hz", params->bpf_high_edge_hz) ||
      !SetFloatInDict(dict, "energy_cutoff_hz", params->energy_cutoff_hz) ||
      !SetFloatInDict(dict, "energy_tau_s", params->energy_tau_s) ||
      !SetFloatInDict(dict, "noise_tau_s", params->noise_tau_s) ||
      !SetFloatInDict(dict, "agc_strength", params->agc_strength) ||
      !SetFloatInDict(dict, "denoise_thresh_factor",
                      params->denoise_thresh_factor) ||
      !SetFloatInDict(dict, "gain_tau_attack_s", params->gain_tau_attack_s) ||
      !SetFloatInDict(dict, "gain_tau_release_s", params->gain_tau_release_s) ||
      !SetFloatInDict(dict, "compressor_exponent",
                      params->compressor_exponent) ||
      !SetFloatInDict(dict, "compressor_delta", params->compressor_delta) ||
      !SetFloatInDict(dict, "output_gain", params->output_gain)) {
    Py_XDECREF(dict);
    return;
  }
  PyModule_AddObject(m, name, dict);
}

PyMODINIT_FUNC PyInit_energy_envelope(void) {
  import_array();
  PyObject* m = PyModule_Create(&kModule);
  kEnergyEnvelopeType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&kEnergyEnvelopeType) >= 0) {
    Py_INCREF(&kEnergyEnvelopeType);
    PyModule_AddObject(m, "EnergyEnvelope", (PyObject*)&kEnergyEnvelopeType);
  }
  AddParamsDict(m, "BASEBAND_PARAMS", &kEnergyEnvelopeBasebandParams);
  AddParamsDict(m, "VOWEL_PARAMS", &kEnergyEnvelopeVowelParams);
  AddParamsDict(m, "SH_FRICATIVE_PARAMS", &kEnergyEnvelopeShFricativeParams);
  AddParamsDict(m, "FRICATIVE_PARAMS", &kEnergyEnvelopeFricativeParams);
  return m;
}
