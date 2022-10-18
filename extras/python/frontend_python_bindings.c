/* Copyright 2019, 2021-2022 Google LLC
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
 * Python bindings for the CARL+PCEN frontend C implementation.
 *
 * These bindings wrap the carl_frontend.c library in frontend as a `frontend`
 * Python module containing a `CarlFrontend` Python class.
 *
 * The interface is as follows. See also carl_frontend_test.py for use example.
 *
 * class CarlFrontend(object):
 *
 *   def __init__(self,
 *                input_sample_rate_hz=16000.0,
 *                block_size=64,
 *                highest_pole_frequency_hz=7000.0,
 *                min_pole_frequency_hz=100.0,
 *                step_erbs=0.5,
 *                envelope_cutoff_hz=20.0,
 *                pcen_time_constant_s=0.3,
 *                pcen_cross_channel_diffusivity=100.0,
 *                pcen_init_value=1e-7,
 *                pcen_alpha=0.7,
 *                pcen_beta=0.2,
 *                pcen_gamma=1e-12,
 *                pcen_delta=0.001)
 *    """Constructor. [Wraps `CarlFrontendMake()` in the C library.]
 *
 *    Args:
 *      The arguments correspond to the CarlFrontendParams in carl_frontend.h in
 *      frontend. See that file for details.
 *    Raises:
 *      ValueError: if parameters are invalid. (In this case, the C library may
 *        write additional details to stderr.)
 *    """
 *
 *  @property
 *  def num_channels(self)
 *    """Number of output channels."""
 *
 *  @property
 *  def block_size(self)
 *    """The block_size."""
 *
 *  def reset(self)
 *    """Resets CarlFrontend to initial state. [Wraps `CarlFrontendReset()`.]"""
 *
 *  def process_samples(self, input_samples)
 *    """Process samples in a streaming manner.
 *
 *    Calls the C function `CarlFrontendProcessSamples()` on each input block.
 *
 *    Args:
 *      input_samples: numpy array. NOTE: Size must be a multiple of block_size.
 *    Returns:
 *      2-D array of shape (len(input_samples)/block_size, num_channels).
 *    """
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
#include "src/frontend/carl_frontend.h"
#include "numpy/arrayobject.h"

typedef struct {
  PyObject_HEAD
  CarlFrontend* frontend;
} CarlFrontendObject;

/* Define `CarlFrontend.__init__`. */
static int CarlFrontendObjectInit(CarlFrontendObject* self, PyObject* args,
                                  PyObject* kw) {
  CarlFrontendParams params = kCarlFrontendDefaultParams;
  static const char* keywords[] = {"input_sample_rate_hz",
                                   "block_size",
                                   "highest_pole_frequency_hz",
                                   "min_pole_frequency_hz",
                                   "step_erbs",
                                   "envelope_cutoff_hz",
                                   "pcen_time_constant_s",
                                   "pcen_cross_channel_diffusivity",
                                   "pcen_init_value",
                                   "pcen_alpha",
                                   "pcen_beta",
                                   "pcen_gamma",
                                   "pcen_delta",
                                   NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kw, "|fifffffffffff:__init__", (char**)keywords,
          &params.input_sample_rate_hz, &params.block_size,
          &params.highest_pole_frequency_hz, &params.min_pole_frequency_hz,
          &params.step_erbs, &params.envelope_cutoff_hz,
          &params.pcen_time_constant_s, &params.pcen_cross_channel_diffusivity,
          &params.pcen_init_value, &params.pcen_alpha, &params.pcen_beta,
          &params.pcen_gamma, &params.pcen_delta)) {
    return -1;  /* PyArg_ParseTupleAndKeywords failed. */
  }

  self->frontend = CarlFrontendMake(&params);
  if (self->frontend == NULL) {
    PyErr_SetString(PyExc_ValueError, "Error making CarlFrontend");
    return -1;
  }
  return 0;
}

static void CarlFrontendDealloc(CarlFrontendObject* self) {
  CarlFrontendFree(self->frontend);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

/* Define `CarlFrontend.reset()`. */
static PyObject* CarlFrontendObjectReset(CarlFrontendObject* self,
                                         PyObject* args, PyObject* kw) {
  CarlFrontendReset(self->frontend);
  Py_INCREF(Py_None);
  return (PyObject*)Py_None;
}

/* Define `CarlFrontend.num_channels` property getter. */
static PyObject* CarlFrontendObjectNumChannels(CarlFrontendObject* self) {
  return PyLong_FromLong(CarlFrontendNumChannels(self->frontend));
}

/* Define `CarlFrontend.block_size` property getter. */
static PyObject* CarlFrontendObjectBlockSize(CarlFrontendObject* self) {
  return PyLong_FromLong(CarlFrontendBlockSize(self->frontend));
}

/* Define `CarlFrontend.process_samples`. */
static PyObject* CarlFrontendObjectProcessSamples(CarlFrontendObject* self,
                                                  PyObject* args,
                                                  PyObject* kw) {
  PyObject* samples_arg = NULL;
  static const char* keywords[] = {"samples", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:process_samples",
                                   (char**)keywords, &samples_arg)) {
    return NULL;  /* PyArg_ParseTupleAndKeywords failed. */
  }

  /* Convert input samples to numpy array with contiguous float32 data. We
   * always make a copy (NPY_ARRAY_ENSURECOPY) since CarlFrontendProcessSamples
   * overwrites its input.
   */
  PyArrayObject* samples = (PyArrayObject*)PyArray_FromAny(
      samples_arg, PyArray_DescrFromType(NPY_FLOAT), 0, 0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_FORCECAST |
          NPY_ARRAY_ENSURECOPY | NPY_ARRAY_DEFAULT,
      NULL);

  if (!samples) {  /* PyArray_DescrFromType failed. */
    /* PyArray_DescrFromType already set an error, so just need to return. */
    return NULL;
  } else if (PyArray_NDIM(samples) != 1) {
    PyErr_SetString(PyExc_ValueError, "expected 1-D array");
    goto fail;
  }

  const int block_size = CarlFrontendBlockSize(self->frontend);
  const int num_channels = CarlFrontendNumChannels(self->frontend);
  const int size = PyArray_SIZE(samples);
  if (size % block_size != 0) {
    PyErr_SetString(PyExc_ValueError,
                    "input size must be a multiple of block_size");
    goto fail;
  }

  /* Create output numpy array. */
  npy_intp output_dims[2];
  output_dims[0] = size / block_size;
  output_dims[1] = num_channels;
  PyArrayObject* output =
      (PyArrayObject*)PyArray_SimpleNew(2, output_dims, NPY_FLOAT);
  if (!output) {  /* PyArray_SimpleNew failed. */
    /* PyArray_SimpleNew already set an error, so clean up and return. */
    goto fail;
  }

  /* Process the samples. */
  float* samples_data = (float*)PyArray_DATA(samples);
  float* output_data = (float*)PyArray_DATA(output);

  /* Release the GIL so that other threads can run while processing. */
  Py_BEGIN_ALLOW_THREADS
  int i;
  for (i = 0; i < output_dims[0]; ++i) {
    CarlFrontendProcessSamples(self->frontend, samples_data, output_data);
    samples_data += block_size;
    output_data += num_channels;
  }
  Py_END_ALLOW_THREADS

  Py_DECREF(samples);
  return (PyObject*)output;

fail:
  Py_XDECREF(samples);
  return NULL;
}

/* CarlFrontend's method functions. */
static PyMethodDef kCarlFrontendMethods[] = {
    {"reset", (PyCFunction)CarlFrontendObjectReset,
     METH_VARARGS | METH_KEYWORDS, "Resets CarlFrontend to initial state."},
    {"process_samples", (PyCFunction)CarlFrontendObjectProcessSamples,
     METH_VARARGS | METH_KEYWORDS, "Processes samples in a streaming manner."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* CarlFrontend's getters (properties). */
static PyGetSetDef kCarlFrontendGetSetDef[] = {
    {"num_channels", (getter)CarlFrontendObjectNumChannels, NULL,
     "Number of output channels."},
    {"block_size", (getter)CarlFrontendObjectBlockSize, NULL,
     "Input block size."},
    {NULL} /* Sentinel */
};

/* Define the CarlFrontend Python type. */
static PyTypeObject kCarlFrontendType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "CarlFrontend",                   /* tp_name */
    sizeof(CarlFrontendObject),       /* tp_basicsize */
    0,                                /* tp_itemsize */
    (destructor)CarlFrontendDealloc,  /* tp_dealloc */
    0,                                /* tp_print */
    0,                                /* tp_getattr */
    0,                                /* tp_setattr */
    0,                                /* tp_compare */
    0,                                /* tp_repr */
    0,                                /* tp_as_number */
    0,                                /* tp_as_sequence */
    0,                                /* tp_as_mapping */
    0,                                /* tp_hash */
    0,                                /* tp_call */
    0,                                /* tp_str */
    0,                                /* tp_getattro */
    0,                                /* tp_setattro */
    0,                                /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,               /* tp_flags */
    "CarlFrontend object",            /* tp_doc */
    0,                                /* tp_traverse */
    0,                                /* tp_clear */
    0,                                /* tp_richcompare */
    0,                                /* tp_weaklistoffset */
    0,                                /* tp_iter */
    0,                                /* tp_iternext */
    kCarlFrontendMethods,             /* tp_methods */
    0,                                /* tp_members */
    kCarlFrontendGetSetDef,           /* tp_getset */
    0,                                /* tp_base */
    0,                                /* tp_dict */
    0,                                /* tp_descr_get */
    0,                                /* tp_descr_set */
    0,                                /* tp_dictoffset */
    (initproc)CarlFrontendObjectInit, /* tp_init */
};

static void DefineCarlFrontendType(PyObject* m) {
  kCarlFrontendType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&kCarlFrontendType) >= 0) {
    Py_INCREF(&kCarlFrontendType);
    PyModule_AddObject(m, "CarlFrontend", (PyObject*)&kCarlFrontendType);
  }
}

/* Module methods. */
static PyMethodDef kModuleMethods[] = {
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Module definition. */
static struct PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "frontend",      /* m_name */
    NULL,            /* m_doc */
    (Py_ssize_t)-1,  /* m_size */
    kModuleMethods,  /* m_methods */
    NULL,            /* m_reload */
    NULL,            /* m_traverse */
    NULL,            /* m_clear */
    NULL,            /* m_free */
};

PyMODINIT_FUNC PyInit_frontend(void) {
  import_array();
  PyObject* m = PyModule_Create(&kModule);
  DefineCarlFrontendType(m);
  return m;
}
