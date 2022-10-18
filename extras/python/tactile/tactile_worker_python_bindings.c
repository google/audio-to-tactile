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
 * Python bindings for TactileWorker C implementation.
 *
 * An implementation choice of the Python interpreter is that while it allows
 * concurrent execution, it does not allow Python code to execute in parallel.
 * This makes it unreliable and probably a bad idea to run real-time audio
 * processing in Python. Furthermore, existing Python audio libraries (e.g.
 * PyAudio, PyGame) don't support the case of differing number of input channels
 * vs. output channels, which is critical to us.
 *
 * To circumvent these problems, these bindings and underlying C library
 * implement a background worker `TactileWorker` that runs a PortAudio stream
 * and runs TactileProcessor on it for real-time audio-to-tactile conversion.
 *
 * Input audio can be taken from either a PortAudio input device (microphone) or
 * from a playback queue of samples. The selected input source may be switched
 * at any time using the `SetMicInput` and `SetPlaybackInput` methods.
 *
 * The interface is as follows.
 *
 * # Constant for the number of tactors.
 * NUM_TACTORS = 10
 *
 * class TactileWorker(object):
 *
 *   def __init__(self,
 *                input_device=None,
 *                output_device=None,
 *                chunk_size=256,
 *                sample_rate_hz=16000,
 *                channels=None,
 *                channel_gains_db=None,
 *                global_gain_db=0.0,
 *                cutoff_hz=500.0,
 *                use_equalizer=True,
 *                mid_gain_db=-10.0,
 *                high_gain_db=-5.5,
 *                post_processing_cutoff_hz=1000.0)
 *    """Constructor. [Wraps `TactileWorkerMake()` in the C library.]
 *
 *    The constructor initializes PortAudio and starts a stream with the
 *    specified devices.
 *
 *    Args:
 *      input_device: String, the PortAudio device (microphone) to take input
 *        audio from, or None for no input device.
 *      output_device: String, the PortAudio device to play tactile output
 *        signals to. If not specified, a list of devices is printed.
 *      chunk_size: Integer, number of frames per audio buffer. Increasing this
 *        value may make audio more reliable, but increase latency.
 *      sample_rate_hz: Integer, audio sample rate in Hz. Note that most devices
 *        support only a few standard rates.
 *      channels: String, a comma-delimited list of sources to configure how
 *        the output channels correspond to tactors.
 *      channel_gains_db: String, a comma-delimited list of gains in dB for each
 *        channel.
 *      global_gain_db: Float, global gain in dB applied to all channels (in
 *        combination with those in channel_gains_db, if specified).
 *      cutoff_hz: Float, cutoff in Hz for energy smoothing filters.
 *      use_equalizer: Boolean, if true, apply perceptual equalizer filter to
 *        tactile output.
 *      mid_gain_db: Float, equalizer mid band gain in dB.
 *      high_gain_db: Float, equalizer high band gain in dB.
 *      post_processing_cutoff_hz: Float, cutoff in Hz of post-processing
 *        lowpass filter.
 *
 *    Raises:
 *      ValueError: if parameters are invalid. The C library may write
 *        additional details to stderr.
 *    """
 *
 *  def play(self, input_samples)
 *    """Enqueues input audio samples. [Wraps C function `TactileWorkerPlay`.]
 *
 *    This function appends `input_samples` audio to the playback queue, which
 *    will get converted to tactile and played to the output device when the
 *    playback input source is selected (with SetPlaybackInput).
 *
 *    Args:
 *      input_samples: 1-D numpy array of audio samples at `sample_rate_hz`.
 *    """
 *
 *  def reset(self)
 *    """Resets tactile processing to initial state."""
 *
 *  def set_mic_input(self)
 *    """Sets the worker to take input from the microphone."""
 *
 *  def set_playback_input(self)
 *    """Sets the worker to take input from the playback queue."""
 *
 *  @property
 *  def remaining_playback_samples(self)
 *    """Number of samples remaining before playback completes."""
 *
 *  @property
 *  def volume_meters(self)
 *    """Gets the current tactor volume levels as size-NUM_TACTORS array."""
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
#include "numpy/arrayobject.h"
#include "extras/python/tactile/tactile_worker.h"
#include "extras/tools/util.h"

typedef struct {
  PyObject_HEAD
  TactileWorker* tactile_worker;
} TactileWorkerObject;

/* Define `TactileWorker.__init__`. */
static int TactileWorkerObjectInit(TactileWorkerObject* self, PyObject* args,
                                   PyObject* kw) {
  TactileWorkerParams params;
  TactileWorkerSetDefaultParams(&params);
  int sample_rate_hz = 16000;
  const char* channels = NULL;
  const char* channel_gains_db = NULL;
  float global_gain_db = 0.0f;
  float mid_gain_db = -10.0f;
  float high_gain_db = -5.5f;
  static const char* keywords[] = {
      "input_device", "output_device", "chunk_size", "sample_rate_hz",
      "channels", "channel_gains_db", "global_gain_db", "cutoff_hz",
      "use_equalizer", "mid_gain_db", "high_gain_db",
      "post_processing_cutoff_hz", NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kw, "|zziiszffpfff:__init__", (char**)keywords,
          &params.input_device,
          &params.output_device, &params.chunk_size, &sample_rate_hz, &channels,
          &channel_gains_db, &global_gain_db,
          &params.tactile_processor_params.enveloper_params.energy_cutoff_hz,
          &params.post_processor_params.use_equalizer,
          &mid_gain_db, &high_gain_db,
          &params.post_processor_params.cutoff_hz)) {
    return -1; /* PyArg_ParseTupleAndKeywords failed. */
  }

  if (!ChannelMapParse(kNumTactors, channels, channel_gains_db,
                       &params.channel_map)) {
    PyErr_SetString(PyExc_ValueError, "Error parsing channel configuration");
    return -1;
  }

  params.tactile_processor_params.frontend_params.input_sample_rate_hz =
      sample_rate_hz;

  params.post_processor_params.mid_gain =
      DecibelsToAmplitudeRatio(mid_gain_db);
  params.post_processor_params.high_gain =
      DecibelsToAmplitudeRatio(high_gain_db);
  params.post_processor_params.gain = DecibelsToAmplitudeRatio(global_gain_db);

  printf("Starting TactileWorker.\n");
  self->tactile_worker = TactileWorkerMake(&params);
  if (self->tactile_worker == NULL) {
    PyErr_SetString(PyExc_ValueError, "Error making TactileWorker");
    return -1;
  }
  return 0;
}

static void TactileWorkerDealloc(TactileWorkerObject* self) {
  TactileWorkerFree(self->tactile_worker);
  Py_TYPE(self)->tp_free((PyObject*)self);
  printf("Stopped TactileWorker.\n");
}

/* Define `TactileWorker.reset`. */
static PyObject* TactileWorkerObjectReset(TactileWorkerObject* self) {
  TactileWorkerReset(self->tactile_worker);
  Py_INCREF(Py_None);
  return Py_None;
}

/* Define `TactileWorker.set_mic_input`. */
static PyObject* TactileWorkerObjectSetMicInput(TactileWorkerObject* self) {
  TactileWorkerSetMicInput(self->tactile_worker);
  Py_INCREF(Py_None);
  return Py_None;
}

/* Define `TactileWorker.set_playback_input`. */
static PyObject* TactileWorkerObjectSetPlaybackInput(
    TactileWorkerObject* self) {
  TactileWorkerSetPlaybackInput(self->tactile_worker);
  Py_INCREF(Py_None);
  return Py_None;
}

/* Define `TactileWorker.play`. */
static PyObject* TactileWorkerObjectPlay(TactileWorkerObject* self,
                                         PyObject* args, PyObject* kw) {
  PyObject* samples_arg = NULL;
  static const char* keywords[] = {"samples", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:play", (char**)keywords,
                                   &samples_arg)) {
    return NULL; /* PyArg_ParseTupleAndKeywords failed. */
  }

  /* Convert input samples to numpy array with contiguous float32 data. */
  PyArrayObject* samples = (PyArrayObject*)PyArray_FromAny(
      samples_arg, PyArray_DescrFromType(NPY_FLOAT), 0, 0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_FORCECAST |
          NPY_ARRAY_DEFAULT,
      NULL);

  if (!samples) { /* PyArray_DescrFromType failed. */
    /* PyArray_DescrFromType already set an error, so just need to return. */
    return NULL;
  } else if (PyArray_NDIM(samples) != 1) {
    PyErr_SetString(PyExc_ValueError, "expected 1-D array");
    Py_DECREF(samples);
    return NULL;
  }

  if (!TactileWorkerPlay(self->tactile_worker, (float*)PyArray_DATA(samples),
                         PyArray_SIZE(samples))) {
    fprintf(stderr, "Error playing samples\n");
  }

  Py_DECREF(samples);
  Py_INCREF(Py_None);
  return Py_None;
}

#if PY_MAJOR_VERSION >= 3
static PyObject* MakePyInteger(int value) { return PyLong_FromLong(value); }
#else
static PyObject* MakePyInteger(int value) { return PyInt_FromLong(value); }
#endif

/* Define `TactileWorker.remaining_playback_samples`. */
static PyObject* TactileWorkerObjectGetRemainingPlaybackSamples(
    TactileWorkerObject* self) {
  return MakePyInteger(
      TactileWorkerGetRemainingPlaybackSamples(self->tactile_worker));
}

/* Define `TactileWorker.volume_meters`. */
static PyObject* TactileWorkerObjectGetVolumeMeters(TactileWorkerObject* self) {
  /* Create output numpy array. */
  npy_intp output_dims[1];
  output_dims[0] = kNumTactors;
  PyArrayObject* output =
      (PyArrayObject*)PyArray_SimpleNew(1, output_dims, NPY_FLOAT);
  if (!output) { /* PyArray_SimpleNew failed. */
    /* PyArray_SimpleNew already set an error, so clean up and return. */
    return NULL;
  }

  TactileWorkerGetVolumeMeters(self->tactile_worker, PyArray_DATA(output));
  return (PyObject*)output;
}

/* TactileWorker's method functions. */
static PyMethodDef kTactileWorkerMethods[] = {
    {"reset", (PyCFunction)TactileWorkerObjectReset,
     METH_NOARGS, "Resets tactile processing to initial state."},
    {"set_mic_input", (PyCFunction)TactileWorkerObjectSetMicInput,
     METH_NOARGS, "Sets microphone as input source."},
    {"set_playback_input", (PyCFunction)TactileWorkerObjectSetPlaybackInput,
     METH_NOARGS, "Sets playback queue as input source."},
    {"play", (PyCFunction)TactileWorkerObjectPlay, METH_VARARGS | METH_KEYWORDS,
     "Appends samples to playback queue."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* TactileWorker's getters (properties). */
static PyGetSetDef kTactileWorkerGetSetDef[] = {
    {"remaining_playback_samples",
     (getter)TactileWorkerObjectGetRemainingPlaybackSamples, NULL,
     "Number of samples remaining before playback completes."},
    {"volume_meters", (getter)TactileWorkerObjectGetVolumeMeters, NULL,
     "Volume meters for each tactor."},
    {NULL} /* Sentinel */
};

/* Define the TactileWorker Python type. */
static PyTypeObject kTactileWorkerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "TactileWorker",                   /* tp_name */
    sizeof(TactileWorkerObject),       /* tp_basicsize */
    0,                                 /* tp_itemsize */
    (destructor)TactileWorkerDealloc,  /* tp_dealloc */
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
    "TactileWorker object",            /* tp_doc */
    0,                                 /* tp_traverse */
    0,                                 /* tp_clear */
    0,                                 /* tp_richcompare */
    0,                                 /* tp_weaklistoffset */
    0,                                 /* tp_iter */
    0,                                 /* tp_iternext */
    kTactileWorkerMethods,             /* tp_methods */
    0,                                 /* tp_members */
    kTactileWorkerGetSetDef,           /* tp_getset */
    0,                                 /* tp_base */
    0,                                 /* tp_dict */
    0,                                 /* tp_descr_get */
    0,                                 /* tp_descr_set */
    0,                                 /* tp_dictoffset */
    (initproc)TactileWorkerObjectInit, /* tp_init */
};

static void InitModule(PyObject* m) {
  PyModule_AddIntConstant(m, "NUM_TACTORS", kNumTactors);

  kTactileWorkerType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&kTactileWorkerType) >= 0) {
    Py_INCREF(&kTactileWorkerType);
    PyModule_AddObject(m, "TactileWorker", (PyObject*)&kTactileWorkerType);
  }
}

/* Module methods. */
static PyMethodDef kModuleMethods[] = {
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Module definition. */
static struct PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "tactile_worker", /* m_name */
    NULL,             /* m_doc */
    (Py_ssize_t)-1,   /* m_size */
    kModuleMethods,   /* m_methods */
    NULL,             /* m_reload */
    NULL,             /* m_traverse */
    NULL,             /* m_clear */
    NULL,             /* m_free */
};

PyMODINIT_FUNC PyInit_tactile_worker(void) {
  import_array();
  PyObject* m = PyModule_Create(&kModule);
  InitModule(m);
  return m;
}
