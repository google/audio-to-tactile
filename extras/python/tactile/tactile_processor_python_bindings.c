/* Copyright 2019, 2021 Google LLC
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
 * Python bindings for TactileProcessor C implementation.
 *
 * These bindings wrap the tactile_processor.c library in tactile as a
 * `tactile_processor` Python module containing a `TactileProcessor` class.
 *
 * The interface is as follows. See also tactile_processor_test.py for examples.
 *
 * # Constant for the number of tactors.
 * NUM_TACTORS = 10
 *
 * class TactileProcessor(object):
 *
 *   def __init__(self,
 *                input_sample_rate_hz=16000.0,
 *                block_size=16,
 *                decimation_factor=1,
 *                cutoff_hz=500.0)
 *    """Constructor. [Wraps `TactileProcessorMake()` in the C library.]
 *
 *    Args:
 *      input_sample_rate_hz: Float, input audio sample rate in Hz.
 *      block_size: Integer, input block size. Must be a power of two.
 *      decimation_factor: Integer, decimation factor after computing the energy
 *        envelope.
 *      cutoff_hz: Float, cutoff in Hz for energy smoothing filters.
 *    Raises:
 *      ValueError: if parameters are invalid. (In this case, the C library may
 *        write additional details to stderr.)
 *    """
 *
 *  def reset():
 *    """Resets to initial state."""
 *
 *  def process_samples(self, input_samples)
 *    """Process samples in a streaming manner.
 *
 *    Calls the C function `TactileProcessorProcessSamples()` on each input
 *    audio block.
 *
 *    Args:
 *      input_samples: 1-D numpy array. Size must be a multiple of block_size.
 *    Returns:
 *      2-D array of shape
 *        (len(input_samples) * decimation_factor / block_size, NUM_TACTORS).
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
 *  def block_size(self)
 *    """The block_size."""
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
#include "src/tactile/tactile_processor.h"
#include "numpy/arrayobject.h"
#include "structmember.h"

typedef struct {
  PyObject_HEAD
  TactileProcessor* tactile_processor;
  float input_sample_rate_hz;
  float output_sample_rate_hz;
  int block_size;
  int decimation_factor;
} TactileProcessorObject;

/* Define `TactileProcessor.__init__`. */
static int TactileProcessorObjectInit(TactileProcessorObject* self,
                                      PyObject* args, PyObject* kw) {
  TactileProcessorParams params;
  TactileProcessorSetDefaultParams(&params);
  float cutoff_hz = 500.0f;
  static const char* keywords[] = {"input_sample_rate_hz",
                                   "block_size",
                                   "decimation_factor",
                                   "cutoff_hz",
                                   NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kw, "|fiif:__init__", (char**)keywords,
          &params.frontend_params.input_sample_rate_hz,
          &params.frontend_params.block_size,
          &params.decimation_factor,
          &cutoff_hz)) {
    return -1;  /* PyArg_ParseTupleAndKeywords failed. */
  }
  params.baseband_channel_params.energy_cutoff_hz = cutoff_hz;
  params.vowel_channel_params.energy_cutoff_hz = cutoff_hz;
  params.sh_fricative_channel_params.energy_cutoff_hz = cutoff_hz;
  params.fricative_channel_params.energy_cutoff_hz = cutoff_hz;

  self->tactile_processor = TactileProcessorMake(&params);
  if (self->tactile_processor == NULL) {
    PyErr_SetString(PyExc_ValueError, "Error making TactileProcessor");
    return -1;
  }
  self->input_sample_rate_hz = params.frontend_params.input_sample_rate_hz;
  self->output_sample_rate_hz =
      params.frontend_params.input_sample_rate_hz / params.decimation_factor;
  self->block_size = params.frontend_params.block_size;
  self->decimation_factor = params.decimation_factor;
  return 0;
}

static void TactileProcessorDealloc(TactileProcessorObject* self) {
  TactileProcessorFree(self->tactile_processor);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

/* Define `TactileProcessor.reset`. */
static PyObject* TactileProcessorObjectReset(TactileProcessorObject* self) {
  TactileProcessorReset(self->tactile_processor);
  Py_INCREF(Py_None);
  return Py_None;
}

/* Define `TactileProcessor.process_samples`. */
static PyObject* TactileProcessorObjectProcessSamples(
    TactileProcessorObject* self, PyObject* args, PyObject* kw) {
  PyObject* samples_arg = NULL;
  static const char* keywords[] = {"samples", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:process_samples",
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
    goto fail;
  }

  const int block_size = self->block_size;
  const int decimation_factor = self->decimation_factor;
  const int size = PyArray_SIZE(samples);
  if (size % block_size != 0) {
    PyErr_SetString(PyExc_ValueError,
                    "input size must be a multiple of block_size");
    goto fail;
  }
  const int tactile_samples_per_block =
      kTactileProcessorNumTactors * (block_size / decimation_factor);

  /* Create output numpy array. */
  npy_intp output_dims[2];
  output_dims[0] = size / decimation_factor;
  output_dims[1] = kTactileProcessorNumTactors;
  PyArrayObject* output =
      (PyArrayObject*)PyArray_SimpleNew(2, output_dims, NPY_FLOAT);
  if (!output) {  /* PyArray_SimpleNew failed. */
    /* PyArray_SimpleNew already set an error, so clean up and return. */
    goto fail;
  }

  /* Process the samples. */
  float* samples_data = (float*)PyArray_DATA(samples);
  float* output_data = (float*)PyArray_DATA(output);
  int start;
  for (start = 0; start < size; start += block_size) {
    TactileProcessorProcessSamples(self->tactile_processor,
        samples_data + start, output_data);
    output_data += tactile_samples_per_block;
  }

  Py_DECREF(samples);
  return (PyObject*)output;

fail:
  Py_XDECREF(samples);
  return NULL;
}

/* TactileProcessor's method functions. */
static PyMethodDef kTactileProcessorMethods[] = {
    {"reset", (PyCFunction)TactileProcessorObjectReset,
     METH_NOARGS, "Resets to initial state."},
    {"process_samples", (PyCFunction)TactileProcessorObjectProcessSamples,
     METH_VARARGS | METH_KEYWORDS, "Processes samples in a streaming manner."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Define `input_sample_rate_hz`, etc. as read-only members. */
static PyMemberDef kTactileProcessorMembers[] = {
  {"input_sample_rate_hz", T_FLOAT,
   offsetof(TactileProcessorObject, input_sample_rate_hz), READONLY, ""},
  {"output_sample_rate_hz", T_FLOAT,
   offsetof(TactileProcessorObject, output_sample_rate_hz), READONLY, ""},
  {"block_size", T_INT,
   offsetof(TactileProcessorObject, block_size), READONLY, ""},
  {"decimation_factor", T_INT,
   offsetof(TactileProcessorObject, decimation_factor), READONLY, ""},
  {NULL}  /* Sentinel. */
};

/* Define the TactileProcessor Python type. */
static PyTypeObject kTactileProcessorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "TactileProcessor",                   /* tp_name */
    sizeof(TactileProcessorObject),       /* tp_basicsize */
    0,                                    /* tp_itemsize */
    (destructor)TactileProcessorDealloc,  /* tp_dealloc */
    0,                                    /* tp_print */
    0,                                    /* tp_getattr */
    0,                                    /* tp_setattr */
    0,                                    /* tp_compare */
    0,                                    /* tp_repr */
    0,                                    /* tp_as_number */
    0,                                    /* tp_as_sequence */
    0,                                    /* tp_as_mapping */
    0,                                    /* tp_hash */
    0,                                    /* tp_call */
    0,                                    /* tp_str */
    0,                                    /* tp_getattro */
    0,                                    /* tp_setattro */
    0,                                    /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                   /* tp_flags */
    "TactileProcessor object",            /* tp_doc */
    0,                                    /* tp_traverse */
    0,                                    /* tp_clear */
    0,                                    /* tp_richcompare */
    0,                                    /* tp_weaklistoffset */
    0,                                    /* tp_iter */
    0,                                    /* tp_iternext */
    kTactileProcessorMethods,             /* tp_methods */
    kTactileProcessorMembers,             /* tp_members */
    0,                                    /* tp_getset */
    0,                                    /* tp_base */
    0,                                    /* tp_dict */
    0,                                    /* tp_descr_get */
    0,                                    /* tp_descr_set */
    0,                                    /* tp_dictoffset */
    (initproc)TactileProcessorObjectInit, /* tp_init */
};

static void InitModule(PyObject* m) {
  PyModule_AddIntConstant(m, "NUM_TACTORS", kTactileProcessorNumTactors);

  kTactileProcessorType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&kTactileProcessorType) >= 0) {
    Py_INCREF(&kTactileProcessorType);
    PyModule_AddObject(m, "TactileProcessor",
        (PyObject*)&kTactileProcessorType);
  }
}

/* Module methods. */
static PyMethodDef kModuleMethods[] = {
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Module definition. */
static struct PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "tactile_processor", /* m_name */
    NULL,                /* m_doc */
    (Py_ssize_t)-1,      /* m_size */
    kModuleMethods,      /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
};

PyMODINIT_FUNC PyInit_tactile_processor(void) {
  import_array();
  PyObject* m = PyModule_Create(&kModule);
  InitModule(m);
  return m;
}
