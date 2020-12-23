/* Copyright 2020 Google LLC
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
 * Python bindings for RationalFactorResampler C implementation.
 *
 * These bindings wrap the resampler.c library in dsp as a
 * `rational_factor_resampler_python_bindings` Python module containing a
 * `ResamplerImpl` class and a `KernelImpl` class.
 *
 * The Python library rational_factor_resampler.py wraps these bindings to give
 * a nicer interface and type annotations.
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
#include "src/dsp/rational_factor_resampler.h"
#include "src/dsp/rational_factor_resampler_kernel.h"
#include "numpy/arrayobject.h"
#include "structmember.h"

typedef struct {
  PyObject_HEAD
  RationalFactorResampler* resampler;
} ResamplerImplObject;

/* Define `ResamplerImpl.__init__`. */
static int ResamplerImplObjectInit(ResamplerImplObject* self,
                                   PyObject* args, PyObject* kw) {
  float input_sample_rate_hz = 0.0f;
  float output_sample_rate_hz = 0.0f;
  int num_channels = 1;
  RationalApproximationOptions rational_approximation_options =
      kRationalApproximationDefaultOptions;
  RationalFactorResamplerOptions options =
      kRationalFactorResamplerDefaultOptions;
  options.rational_approximation_options = &rational_approximation_options;
  int max_input_size = 1024;
  static const char* keywords[] = {
      "input_sample_rate_hz",
      "output_sample_rate_hz",
      "num_channels",
      "max_denominator",
      "rational_approximation_max_terms",
      "rational_approximation_convergence_tolerance",
      "filter_radius_factor",
      "cutoff_proportion",
      "kaiser_beta",
      "max_input_size",
      NULL};

  /* The string below defines argument C types to parse from `args` and `kw`
   * ('f' => float, 'd' => double, 'i' => int). The ":__init__" at the end gives
   * the function name to use in error messages.
   * [https://docs.python.org/3/c-api/arg.html]
   */
  if (!PyArg_ParseTupleAndKeywords(
          args, kw, "ffiiidfffi:__init__", (char**)keywords,
          &input_sample_rate_hz,
          &output_sample_rate_hz,
          &num_channels,
          &options.max_denominator,
          &rational_approximation_options.max_terms,
          &rational_approximation_options.convergence_tolerance,
          &options.filter_radius_factor,
          &options.cutoff_proportion,
          &options.kaiser_beta,
          &max_input_size)) {
    return -1;  /* PyArg_ParseTupleAndKeywords failed. */
  }

  self->resampler = RationalFactorResamplerMake(input_sample_rate_hz,
                                                output_sample_rate_hz,
                                                num_channels,
                                                max_input_size,
                                                &options);
  if (self->resampler == NULL) {
    PyErr_SetString(PyExc_ValueError, "Error making Resampler");
    return -1;
  }
  return 0;
}

static void ResamplerImplDealloc(ResamplerImplObject* self) {
  RationalFactorResamplerFree(self->resampler);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

/* Define `ResamplerImpl.reset`. */
static PyObject* ResamplerImplObjectReset(ResamplerImplObject* self) {
  RationalFactorResamplerReset(self->resampler);
  Py_INCREF(Py_None);
  return Py_None;
}

/* Define `ResamplerImpl.process_samples`. */
static PyObject* ResamplerImplObjectProcessSamples(
    ResamplerImplObject* self, PyObject* args, PyObject* kw) {
  PyObject* samples_arg = NULL;
  static const char* keywords[] = {"samples", NULL};

  /* "O:process_samples" => parse a single required arg as a PyObject. */
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
    goto fail;
  }

  RationalFactorResampler* resampler = self->resampler;
  const int num_channels = RationalFactorResamplerNumChannels(resampler);

  if (!((num_channels == 1 && PyArray_NDIM(samples) == 1) ||
        (PyArray_NDIM(samples) == 2 &&
         PyArray_DIM(samples, 1) == num_channels))) {
    PyErr_Format(PyExc_ValueError, "expected array of shape (N, %d)",
                 num_channels);
    goto fail;
  }

  const int num_input_frames = PyArray_DIM(samples, 0);

  /* Create output numpy array. */
  npy_intp output_dims[2];
  output_dims[0] = RationalFactorResamplerNextNumOutputFrames(
      resampler, num_input_frames);
  output_dims[1] = num_channels;
  PyArrayObject* output =
      (PyArrayObject*)PyArray_SimpleNew(
          PyArray_NDIM(samples), output_dims, NPY_FLOAT);
  if (!output) {  /* PyArray_SimpleNew failed. */
    /* PyArray_SimpleNew already set an error, so clean up and return. */
    goto fail;
  }

  /* Process the samples. */
  const int max_input_frames = RationalFactorResamplerMaxInputFrames(resampler);
  const float* samples_data = (const float*)PyArray_DATA(samples);
  float* output_data = (float*)PyArray_DATA(output);
  int start;
  for (start = 0; start < num_input_frames; start += max_input_frames) {
    int block_input_frames = num_input_frames - start;
    if (max_input_frames < block_input_frames) {
      block_input_frames = max_input_frames;
    }
    const int block_output_frames = RationalFactorResamplerProcessSamples(
        resampler, samples_data + start * num_channels, block_input_frames);
    const float* block_output = RationalFactorResamplerOutput(resampler);
    memcpy(output_data, block_output,
           sizeof(float) * block_output_frames * num_channels);
    output_data += block_output_frames * num_channels;
  }

  Py_DECREF(samples);
  return (PyObject*)output;

fail: /* If something went wrong above, we jump here and clean up. */
  Py_XDECREF(samples);
  return NULL;
}

/* Define `ResamplerImpl.rational_factor` property getter. */
static PyObject* ResamplerImplObjectRationalFactor(ResamplerImplObject* self) {
  int factor_numerator;
  int factor_denominator;
  RationalFactorResamplerGetRationalFactor(
      self->resampler, &factor_numerator, &factor_denominator);
  return Py_BuildValue("(ii)", factor_numerator, factor_denominator);
}

/* Define `ResamplerImpl.num_channels` property getter. */
static PyObject* ResamplerImplObjectNumChannels(ResamplerImplObject* self) {
  return Py_BuildValue("i",
                       RationalFactorResamplerNumChannels(self->resampler));
}

/* Define `ResamplerImpl.flush_frames` property getter. */
static PyObject* ResamplerImplObjectFlushFrames(ResamplerImplObject* self) {
  return Py_BuildValue("i",
                       RationalFactorResamplerFlushFrames(self->resampler));
}

/* Resampler's method functions. */
static PyMethodDef kResamplerImplMethods[] = {
    {"reset", (PyCFunction)ResamplerImplObjectReset,
     METH_NOARGS, "Resets to initial state."},
    {"process_samples", (PyCFunction)ResamplerImplObjectProcessSamples,
     METH_VARARGS | METH_KEYWORDS, "Processes samples in a streaming manner."},
    {NULL} /* Sentinel */
};

/* Resamplers's getters (properties). */
static PyGetSetDef kResamplerImplGetSetDef[] = {
    {"rational_factor", (getter)ResamplerImplObjectRationalFactor,
     NULL, "Rational resampling factor."},
    {"num_channels", (getter)ResamplerImplObjectNumChannels,
     NULL, "Number of channels."},
    {"flush_frames", (getter)ResamplerImplObjectFlushFrames,
     NULL, "Number of input frames needed to flush the reampler."},
    {NULL} /* Sentinel */
};

/* Define the ResamplerImpl Python type. For meanings of these fields, see
 * https://docs.python.org/3/c-api/typeobj.html
 */
static PyTypeObject kResamplerImplType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "ResamplerImpl",                   /* tp_name */
    sizeof(ResamplerImplObject),       /* tp_basicsize */
    0,                                 /* tp_itemsize */
    (destructor)ResamplerImplDealloc,  /* tp_dealloc */
    0,                                 /* tp_print */
    0,                                 /* tp_getattr */
    0,                                 /* tp_setattr */
    0,                                 /* tp_compare */
    0,                                 /* tp_repr */
    0,                                 /* tp_as_number */
    0,                                 /* tp_as_sequence */
    0,                                 /* tp_as_mapping */
    0,                                 /* tp_hash */
    0,                                 /* tp_call */
    0,                                 /* tp_str */
    0,                                 /* tp_getattro */
    0,                                 /* tp_setattro */
    0,                                 /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                /* tp_flags */
    "ResamplerImpl object",            /* tp_doc */
    0,                                 /* tp_traverse */
    0,                                 /* tp_clear */
    0,                                 /* tp_richcompare */
    0,                                 /* tp_weaklistoffset */
    0,                                 /* tp_iter */
    0,                                 /* tp_iternext */
    kResamplerImplMethods,             /* tp_methods */
    0,                                 /* tp_members */
    kResamplerImplGetSetDef,           /* tp_getset */
    0,                                 /* tp_base */
    0,                                 /* tp_dict */
    0,                                 /* tp_descr_get */
    0,                                 /* tp_descr_set */
    0,                                 /* tp_dictoffset */
    (initproc)ResamplerImplObjectInit, /* tp_init */
};

typedef struct {
  PyObject_HEAD
  RationalFactorResamplerKernel kernel;
} KernelImplObject;

/* Define `factor` and `radius` as read-only members. */
static PyMemberDef kKernelImplMembers[] = {
  {"factor", T_DOUBLE, offsetof(KernelImplObject, kernel) +
     offsetof(RationalFactorResamplerKernel, factor), READONLY, ""},
  {"radius", T_DOUBLE, offsetof(KernelImplObject, kernel) +
     offsetof(RationalFactorResamplerKernel, radius), READONLY, ""},
  {"radians_per_sample", T_DOUBLE, offsetof(KernelImplObject, kernel) +
     offsetof(RationalFactorResamplerKernel, radians_per_sample), READONLY, ""},
  {NULL}  /* Sentinel. */
};

/* Define `Kernel.__init__`. */
static int KernelImplObjectInit(KernelImplObject* self,
                                PyObject* args, PyObject* kw) {
  float input_sample_rate_hz = 0.0f;
  float output_sample_rate_hz = 0.0f;
  RationalFactorResamplerOptions options =
      kRationalFactorResamplerDefaultOptions;
  static const char* keywords[] = {"input_sample_rate_hz",
                                   "output_sample_rate_hz",
                                   "filter_radius_factor",
                                   "cutoff_proportion",
                                   "kaiser_beta",
                                   NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kw, "fffff:__init__", (char**)keywords,
          &input_sample_rate_hz,
          &output_sample_rate_hz,
          &options.filter_radius_factor,
          &options.cutoff_proportion,
          &options.kaiser_beta)) {
    return -1;  /* PyArg_ParseTupleAndKeywords failed. */
  }

  if (!RationalFactorResamplerKernelInit(&self->kernel,
                                         input_sample_rate_hz,
                                         output_sample_rate_hz,
                                         options.filter_radius_factor,
                                         options.cutoff_proportion,
                                         options.kaiser_beta)) {
    PyErr_SetString(PyExc_ValueError, "Error making Kernel");
    return -1;
  }
  return 0;
}

static void KernelImplDealloc(KernelImplObject* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

/* Define `Kernel.__call__`. */
static PyObject* KernelImplObjectCall(KernelImplObject* self,
                                      PyObject* args, PyObject* kw) {
  PyObject* samples_arg = NULL;
  static const char* keywords[] = {"samples", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:__call__",
                                   (char**)keywords, &samples_arg)) {
    return NULL;  /* PyArg_ParseTupleAndKeywords failed. */
  }

  /* Convert input to numpy array with contiguous float64 data. */
  PyArrayObject* x = (PyArrayObject*)PyArray_FromAny(
      samples_arg, PyArray_DescrFromType(NPY_DOUBLE), 0, 0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_FORCECAST |
          NPY_ARRAY_DEFAULT,
      NULL);

  if (!x) {  /* PyArray_DescrFromType failed. */
    /* PyArray_DescrFromType already set an error, so just need to return. */
    return NULL;
  }


  /* Create output numpy array. */
  PyArrayObject* output =
      (PyArrayObject*)PyArray_SimpleNew(
          PyArray_NDIM(x), PyArray_DIMS(x), NPY_DOUBLE);
  if (!output) {  /* PyArray_SimpleNew failed. */
    /* PyArray_SimpleNew already set an error, so clean up and return. */
    Py_XDECREF(x);
    return NULL;
  }

  /* Evaluate the kernel. */
  const double* x_data = (const double*)PyArray_DATA(x);
  double* output_data = (double*)PyArray_DATA(output);
  const int size = PyArray_SIZE(x);
  int i;
  for (i = 0; i < size; ++i) {
    output_data[i] = RationalFactorResamplerKernelEval(
        &self->kernel, x_data[i]);
  }

  Py_DECREF(x);
  return (PyObject*)output;
}

/* Define the Kernel Python type. */
static PyTypeObject kKernelImplType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "KernelImpl",                      /* tp_name */
    sizeof(KernelImplObject),          /* tp_basicsize */
    0,                                 /* tp_itemsize */
    (destructor)KernelImplDealloc,     /* tp_dealloc */
    0,                                 /* tp_print */
    0,                                 /* tp_getattr */
    0,                                 /* tp_setattr */
    0,                                 /* tp_compare */
    0,                                 /* tp_repr */
    0,                                 /* tp_as_number */
    0,                                 /* tp_as_sequence */
    0,                                 /* tp_as_mapping */
    0,                                 /* tp_hash */
    (ternaryfunc)KernelImplObjectCall, /* tp_call */
    0,                                 /* tp_str */
    0,                                 /* tp_getattro */
    0,                                 /* tp_setattro */
    0,                                 /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                /* tp_flags */
    "KernelImpl object",               /* tp_doc */
    0,                                 /* tp_traverse */
    0,                                 /* tp_clear */
    0,                                 /* tp_richcompare */
    0,                                 /* tp_weaklistoffset */
    0,                                 /* tp_iter */
    0,                                 /* tp_iternext */
    0,                                 /* tp_methods */
    kKernelImplMembers,                /* tp_members */
    0,                                 /* tp_getset */
    0,                                 /* tp_base */
    0,                                 /* tp_dict */
    0,                                 /* tp_descr_get */
    0,                                 /* tp_descr_set */
    0,                                 /* tp_dictoffset */
    (initproc)KernelImplObjectInit,    /* tp_init */
};

/* Add `type_object` to module in either Python 2 or Python 3, following
 * https://docs.python.org/2.7/extending/newtypes.html
 * https://docs.python.org/3/extending/newtypes_tutorial.html
 */
static /*bool*/ int AddTypeToModule(PyObject* m, PyTypeObject* type_object) {
  /* Set __new__ to the default implementation. We would set it in the struct
   * initializers above, but as the 2.7 documentation notes, "On some platforms
   * or compilers, we can't statically initialize a structure member with a
   * function defined in another C module, so, instead, we'll assign the tp_new
   * slot in the module initialization function".
   */
  type_object->tp_new = PyType_GenericNew;
  /* "Finalize" the type definition. */
  if (PyType_Ready(type_object) < 0) { return 0; }

  Py_INCREF(type_object);
  /* Add the type to the module dictionary.
   * NOTE: PyModule_AddObject() can fail in principle, but only does so on
   * programming error (e.g. `m` is not a module). Most code, for instance
   * itertools and protobufs, doesn't check PyModule_AddObject's return value.
   */
  PyModule_AddObject(m, type_object->tp_name, (PyObject*)type_object);
  return 1;
}

/* Module definition. */
static struct PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "rational_factor_resampler_python_bindings", /* m_name */
    NULL,                                        /* m_doc */
    (Py_ssize_t)-1,                              /* m_size */
};

PyMODINIT_FUNC PyInit_rational_factor_resampler_python_bindings() {
  import_array();
  PyObject* m = PyModule_Create(&kModule);
  if (m == NULL) { return NULL; }

  if (!AddTypeToModule(m, &kResamplerImplType) ||
      !AddTypeToModule(m, &kKernelImplType)) {
    Py_DECREF(m);
    return NULL;
  }
  return m;
}
