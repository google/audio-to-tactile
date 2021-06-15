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
 * Python bindings for phoneme classifier inference C implementation.
 *
 * These bindings wrap the classify_phoneme.c library in
 * audio/tactile/phone_embedding as a `classify_phoneme` Python module to
 * perform phoneme classification from consecutive CARL+PCEN frames. The
 * CarlFrontend must run with the parameters:
 *
 *   input_sample_rate_hz = 16000 Hz
 *   block_size = 128 samples (8ms)
 *   pcen_cross_channel_diffusivity = 60.0
 *
 * and otherwise default parameters. There is an example use in the unit test
 * classify_phoneme_test.py.
 *
 * The interface is as follows.
 *
 * def classify_phoneme_labels(frames):
 *   """Gets classification labels. [Wraps `ClassifyPhone()` in the C library.]
 *
 *   Args:
 *     frames: 2D array of shape [NUM_FRAMES, NUM_CHANNELS].
 *   Returns:
 *     dict of classification labels:
 *
 *     {
 *       'phoneme': Phoneme label as an ARPABET code (e.g., 'ch').
 *       'manner': Manner label.
 *       'place': Place label.
 *       'vad': Bool, true => speech, false => silence.
 *       'vowel': Bool, true => vowel, false => consonant.
 *       'diphthong': Bool, true => diphthong, false => monophthong.
 *       'lax_vowel': Bool, true => lax vowel, false => tense vowel.
 *       'voice': Bool, true => voiced, fales => unvoice.
 *     }
 *
 * def classify_phoneme_scores(frames):
 *   """Gets classification scores. [Wraps `ClassifyPhone()` in the C library.]
 *
 *   Args:
 *     frames: 2D array of shape [NUM_FRAMES, NUM_CHANNELS].
 *   Returns:
 *     dict of classification scores between 0.0 and 1.0, with higher score
 *     implying greater confidence:
 *
 *     {
 *       'phoneme': {  Fine-grained classification, keyed by ARPABET code.
 *         'sil': silence score,
 *         'aa': AA score,
 *         ...
 *       },
 *       'manner': {  Manner, supposing the phoneme is a consonant.
 *         'nasal': nasal score,
 *         'stop': stop score,
 *         'affricate': affricate score,
 *         'fricative': fricative score,
 *         'approximant': approximant score,
 *       },
 *       'place': {  Place, supposing the phoneme is a consonant.
 *         'front': front score,
 *         'middle': middle score,
 *         'back': back score,
 *       },
 *       'vad': voice activity detection (VAD) score,
 *       'vowel': consonant/vowel score,
 *       'diphthong': monophthong/diphthong score, supposing phoneme is a vowel,
 *       'lax_vowel': tense/lax score, supposing phoneme is a vowel,
 *       'voiced': voiced score,
 *    }
 *   """
 *
 * The expected number of frames and CARL channels are exposed as a module
 * constants `NUM_FRAMES` and `NUM_CHANNELS`.
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
#include "numpy/arrayobject.h"
#include "src/phonetics/classify_phoneme.h"

#define NUM_FRAMES kClassifyPhonemeNumFrames
#define NUM_CHANNELS kClassifyPhonemeNumChannels

/* Convert input arg to numpy array with contiguous float32 data. The caller
 * must call Py_XDECREF on the returned object to release memory.
 */
static PyArrayObject* GetInputFramesArray(PyObject* frames_arg) {
  PyArrayObject* frames = (PyArrayObject*)PyArray_FromAny(
      frames_arg, PyArray_DescrFromType(NPY_FLOAT), 0, 0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_FORCECAST |
          NPY_ARRAY_DEFAULT,
      NULL);

  if (!frames) {  /* PyArray_DescrFromType failed. */
    /* PyArray_DescrFromType already set an error, so just need to return. */
    return NULL;
  } else if (PyArray_NDIM(frames) != 2) {
    PyErr_Format(PyExc_ValueError,
        "input must be a 2D array of shape (%d, %d), got: %dD array",
        NUM_FRAMES, NUM_CHANNELS, PyArray_NDIM(frames));
    Py_DECREF(frames);
    return NULL;
  }

  const int num_frames = PyArray_DIM(frames, 0);
  const int num_channels = PyArray_DIM(frames, 1);
  if (num_frames != NUM_FRAMES || num_channels != NUM_CHANNELS) {
    Py_DECREF(frames);
    PyErr_Format(
        PyExc_ValueError,
        "input must be a 2D array of shape (%d, %d), got: (%d, %d)",
        NUM_FRAMES, NUM_CHANNELS, num_frames, num_channels);
    return NULL;
  }
  return frames;
}

/* Helper functions for putting things into Python dictionaries. */

/* Sets a bool value in `dict`. Returns 1 on success, 0 on failure. */
static int SetBoolInDict(PyObject* dict, const char* key, /*bool*/ int value) {
  PyObject* value_obj = value ? Py_True : Py_False;
  const int success = (PyDict_SetItemString(dict, key, value_obj) == 0);
  return success;
}

/* Sets a string value in `dict`. Returns 1 on success, 0 on failure. */
static int SetStringInDict(PyObject* dict, const char* key, const char* value) {
  PyObject* value_obj = PyUnicode_FromString(value);
  if (!value_obj) { return 0; }
  const int success = (PyDict_SetItemString(dict, key, value_obj) == 0);
  /* PyDict_SetItemString() does *not* steal a reference to value_obj. */
  Py_DECREF(value_obj);
  return success;
}

/* Sets a float value in `dict`. Returns 1 on success, 0 on failure. */
static int SetFloatInDict(PyObject* dict, const char* key, float value) {
  PyObject* value_obj = PyFloat_FromDouble(value);
  if (!value_obj) { return 0; }
  const int success = (PyDict_SetItemString(dict, key, value_obj) == 0);
  /* PyDict_SetItemString() does *not* steal a reference to value_obj. */
  Py_DECREF(value_obj);
  return success;
}

/* Sets a subdictionary in `parent_dict`, populated with keys and scores:
 *
 *   parent_dict[subdict_name] = dict(zip(keys, scores))
 *
 * Returns 1 on suceess, 0 on failure.
 */
static int SetScoresSubDict(PyObject* parent_dict,
                            const char* subdict_name,
                            const char** keys,
                            const float* scores,
                            int size) {
  PyObject* subdict = PyDict_New();
  if (!subdict) { return 0; }

  int i;
  for (i = 0; i < size; ++i) {  /* Populate subdict with keys and scores. */
    if (!SetFloatInDict(subdict, keys[i], scores[i])) {
      Py_DECREF(subdict);
      return 0;
    }
  }

  /* Set `subdict` in `parent_dict`. */
  const int success =
    (PyDict_SetItemString(parent_dict, subdict_name, subdict) == 0);
  Py_DECREF(subdict);
  return success;
}

/* Define `classify_phone_labels()`. */
static PyObject* ClassifyPhonemeLabelsPython(
    PyObject* dummy, PyObject* args, PyObject* kw) {
  PyObject* frames_arg;
  static const char* keywords[] = {"frames", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:classify_phoneme_labels",
                                   (char**)keywords, &frames_arg)) {
    return NULL;  /* PyArg_ParseTupleAndKeywords failed. */
  }

  PyArrayObject* frames;
  if (!(frames = GetInputFramesArray(frames_arg))) {
    return NULL;
  }

  const float* frames_data = (const float*)PyArray_DATA(frames);
  ClassifyPhonemeLabels labels;
  ClassifyPhoneme(frames_data, &labels, NULL);
  Py_DECREF(frames);

  PyObject* dict = PyDict_New();
  if (!dict ||
      !SetStringInDict(dict, "phoneme",
                       kClassifyPhonemePhonemeNames[labels.phoneme]) ||
      !SetStringInDict(dict, "manner",
                       kClassifyPhonemeMannerNames[labels.manner]) ||
      !SetStringInDict(dict, "place",
                       kClassifyPhonemePlaceNames[labels.place]) ||
      !SetBoolInDict(dict, "vad", labels.vad) ||
      !SetBoolInDict(dict, "vowel", labels.vowel) ||
      !SetBoolInDict(dict, "diphthong", labels.diphthong) ||
      !SetBoolInDict(dict, "lax_vowel", labels.lax_vowel) ||
      !SetBoolInDict(dict, "voiced", labels.voiced)) {
    Py_XDECREF(dict);
    return NULL;
  }

  return dict;
}

/* Define `classify_phone_scores()`. */
static PyObject* ClassifyPhonemeScoresPython(
    PyObject* dummy, PyObject* args, PyObject* kw) {
  PyObject* frames_arg;
  static const char* keywords[] = {"frames", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:classify_phoneme_scores",
                                   (char**)keywords, &frames_arg)) {
    return NULL;  /* PyArg_ParseTupleAndKeywords failed. */
  }

  PyArrayObject* frames;
  if (!(frames = GetInputFramesArray(frames_arg))) {
    return NULL;
  }

  const float* frames_data = (const float*)PyArray_DATA(frames);
  ClassifyPhonemeScores scores;
  ClassifyPhoneme(frames_data, NULL, &scores);
  Py_DECREF(frames);

  PyObject* dict = PyDict_New();
  if (!dict ||
      !SetScoresSubDict(dict, "phoneme", kClassifyPhonemePhonemeNames,
                        scores.phoneme, kClassifyPhonemeNumPhonemes) ||
      !SetScoresSubDict(dict, "manner", kClassifyPhonemeMannerNames,
                        scores.manner, kClassifyPhonemeNumManners) ||
      !SetScoresSubDict(dict, "place", kClassifyPhonemePlaceNames,
                        scores.place, kClassifyPhonemeNumPlaces) ||
      !SetFloatInDict(dict, "vad", scores.vad) ||
      !SetFloatInDict(dict, "vowel", scores.vowel) ||
      !SetFloatInDict(dict, "diphthong", scores.diphthong) ||
      !SetFloatInDict(dict, "lax_vowel", scores.lax_vowel) ||
      !SetFloatInDict(dict, "voiced", scores.voiced)) {
    Py_XDECREF(dict);
    return NULL;
  }

  return dict;
}

/* Module methods. */
static PyMethodDef kModuleMethods[] = {
    {"classify_phoneme_labels", (PyCFunction)ClassifyPhonemeLabelsPython,
      METH_VARARGS | METH_KEYWORDS, "Gets phoneme classification labels."},
    {"classify_phoneme_scores", (PyCFunction)ClassifyPhonemeScoresPython,
      METH_VARARGS | METH_KEYWORDS, "Gets phoneme classification scores."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Module definition. */
static struct PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "classify_phoneme",  /* m_name */
    NULL,                /* m_doc */
    (Py_ssize_t)-1,      /* m_size */
    kModuleMethods,      /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
};

static PyObject* MakeListOfStrings(const char** names, int size) {
  PyObject* list = PyList_New(size);
  if (!list) { return NULL; }
  int i;
  for (i = 0; i < size; ++i) {
    PyObject* name = PyUnicode_FromString(names[i]);
    if (PyList_SetItem(list, i, name)) {
      Py_XDECREF(name);
      Py_XDECREF(list);
      return NULL;
    }
  }
  return list;
}

static void InitModule(PyObject* m) {
  /* Make `NUM_FRAMES` module int constant. */
  PyModule_AddIntMacro(m, NUM_FRAMES);
  /* Make `NUM_CHANNELS` module int constant. */
  PyModule_AddIntMacro(m, NUM_CHANNELS);

  /* Make `PHONEMES` name list. */
  PyObject* phonemes = MakeListOfStrings(kClassifyPhonemePhonemeNames,
                                         kClassifyPhonemeNumPhonemes);
  if (!phonemes) { return; }
  PyModule_AddObject(m, "PHONEMES", phonemes);

  /* Make `MANNERS` name list. */
  PyObject* manners = MakeListOfStrings(kClassifyPhonemeMannerNames,
                                        kClassifyPhonemeNumManners);
  if (!manners) { return; }
  PyModule_AddObject(m, "MANNERS", manners);

  /* Make `PLACES` name list. */
  PyObject* places = MakeListOfStrings(kClassifyPhonemePlaceNames,
                                       kClassifyPhonemeNumPlaces);
  if (!places) { return; }
  PyModule_AddObject(m, "PLACES", places);
}

PyMODINIT_FUNC PyInit_classify_phoneme(void) {
  import_array();
  PyObject* m = PyModule_Create(&kModule);
  if (m) {
    InitModule(m);
  }
  return m;
}
