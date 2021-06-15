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
 * Python bindings for PostProcessor C implementation.
 *
 * These bindings wrap the post_processor.c library in tactile as a
 * `post_processor` Python module containing an `PostProcessor` class.
 *
 * The interface is as follows.
 *
 * class PostProcessor(object):
 *
 *   def __init__(self,
 *                sample_rate_hz,
 *                num_channels,
 *                use_equalizer=True,
 *                mid_gain=0.31623,
 *                high_gain=0.53088,
 *                gain=1.0,
 *                max_amplitude=0.96,
 *                cutoff_hz=1000.0)
 *    """Constructor. [Wraps `PostProcessorInit()` in the C library.]"""
 *
 *  def reset():
 *    """Resets to initial state."""
 *
 *  def process_samples(self, input_samples)
 *    """Process samples in a streaming manner.
 *
 *    Args:
 *      input_samples: 1-D numpy array.
 *    Returns:
 *      Array of output samples of the same size.
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

#include "Python.h"
/* Disallow Numpy 1.7 deprecated symbols. */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "src/tactile/post_processor.h"
#include "numpy/arrayobject.h"
#include "structmember.h"

typedef struct {
  PyObject_HEAD
  PostProcessor post_processor;
} PostProcessorObject;

/* Define `PostProcessor.__init__`. */
static int PostProcessorObjectInit(PostProcessorObject* self,
                                   PyObject* args, PyObject* kw) {
  PostProcessorParams params;
  PostProcessorSetDefaultParams(&params);
  float sample_rate_hz;
  int num_channels;
  static const char* keywords[] = {"sample_rate_hz",
                                   "num_channels",
                                   "use_equalizer",
                                   "mid_gain",
                                   "high_gain",
                                   "gain",
                                   "max_amplitude",
                                   "cutoff_hz",
                                   NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kw, "fi|pfffff:__init__", (char**)keywords,
          &sample_rate_hz,
          &num_channels,
          &params.use_equalizer,
          &params.mid_gain,
          &params.high_gain,
          &params.gain,
          &params.max_amplitude,
          &params.cutoff_hz)) {
    return -1;  /* PyArg_ParseTupleAndKeywords failed. */
  }

  if (!PostProcessorInit(&self->post_processor, &params,
                         sample_rate_hz, num_channels)) {
    PyErr_SetString(PyExc_ValueError, "Error making PostProcessor");
    return -1;
  }
  return 0;
}

static void PostProcessorDealloc(PostProcessorObject* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

/* Define `PostProcessor.reset`. */
static PyObject* PostProcessorObjectReset(PostProcessorObject* self) {
  PostProcessorReset(&self->post_processor);
  Py_INCREF(Py_None);
  return Py_None;
}

/* Define `PostProcessor.process_samples`. */
static PyObject* PostProcessorObjectProcessSamples(
    PostProcessorObject* self, PyObject* args, PyObject* kw) {
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
          NPY_ARRAY_DEFAULT | NPY_ARRAY_ENSURECOPY,
      NULL);

  if (!samples) {  /* PyArray_DescrFromType failed. */
    /* PyArray_DescrFromType already set an error, so just need to return. */
    return NULL;
  } else if (PyArray_NDIM(samples) != 2) {
    PyErr_SetString(PyExc_ValueError, "expected 2-D array");
    Py_DECREF(samples);
    return NULL;
  } else if (PyArray_DIM(samples, 1) != self->post_processor.num_channels) {
    return PyErr_Format(
        PyExc_ValueError, "expected array with %d columns, got: %d",
        self->post_processor.num_channels, PyArray_DIM(samples, 1));
    Py_DECREF(samples);
    return NULL;
  }

  /* Process the samples. */
  float* samples_data = (float*)PyArray_DATA(samples);
  const int num_frames = PyArray_DIM(samples, 0);
  PostProcessorProcessSamples(
      &self->post_processor, samples_data, num_frames);

  return (PyObject*)samples;
}

/* PostProcessor's method functions. */
static PyMethodDef kPostProcessorMethods[] = {
    {"reset", (PyCFunction)PostProcessorObjectReset,
     METH_NOARGS, "Resets to initial state."},
    {"process_samples", (PyCFunction)PostProcessorObjectProcessSamples,
     METH_VARARGS | METH_KEYWORDS, "Processes samples in a streaming manner."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Define the PostProcessor Python type. */
static PyTypeObject kPostProcessorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "PostProcessor",                    /* tp_name */
    sizeof(PostProcessorObject),        /* tp_basicsize */
    0,                                  /* tp_itemsize */
    (destructor)PostProcessorDealloc,   /* tp_dealloc */
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
    "PostProcessor object",             /* tp_doc */
    0,                                  /* tp_traverse */
    0,                                  /* tp_clear */
    0,                                  /* tp_richcompare */
    0,                                  /* tp_weaklistoffset */
    0,                                  /* tp_iter */
    0,                                  /* tp_iternext */
    kPostProcessorMethods,              /* tp_methods */
    0,                                  /* tp_members */
    0,                                  /* tp_getset */
    0,                                  /* tp_base */
    0,                                  /* tp_dict */
    0,                                  /* tp_descr_get */
    0,                                  /* tp_descr_set */
    0,                                  /* tp_dictoffset */
    (initproc)PostProcessorObjectInit,  /* tp_init */
};

/* Module methods. */
static PyMethodDef kModuleMethods[] = {
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Module definition. */
static struct PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "post_processor",  /* m_name */
    NULL,               /* m_doc */
    (Py_ssize_t)-1,     /* m_size */
    kModuleMethods,     /* m_methods */
    NULL,               /* m_reload */
    NULL,               /* m_traverse */
    NULL,               /* m_clear */
    NULL,               /* m_free */
};

PyMODINIT_FUNC PyInit_post_processor(void) {
  import_array();
  PyObject* m = PyModule_Create(&kModule);
  kPostProcessorType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&kPostProcessorType) >= 0) {
    Py_INCREF(&kPostProcessorType);
    PyModule_AddObject(m, "PostProcessor", (PyObject*)&kPostProcessorType);
  }
  return m;
}
