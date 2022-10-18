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
 * Python bindings for WAV reading and writing.
 *
 * These bindings wrap the read_wav_file_generic.c and write_wav_file_generic.c
 * libraries in dsp as a `wav_io` Python module implementing the following two
 * functions:
 *
 * def read_wav_impl(file_object, dtype=None)
 *   """Reads a WAV audio from file object.
 *
 *   Args:
 *     file_object: Open file-like object.
 *     dtype: Numpy dtype, one of np.int16, np.int32, or np.float32 to convert
 *       the read samples to. Or None to return samples without conversion.
 *   Returns:
 *     (samples, sample_rate_hz) 2-tuple where `sample_rate_hz` is the sample
 *     rate in Hz and `samples` is an array of shape [num_frames, num_channels]
 *     and dtype according to the encoding used in the file.
 *
 *
 * def write_wav_impl(file_object, samples, sample_rate_hz)
 *   """Writes WAV audio to a file object.
 *
 *   Args:
 *     file_object: Open writeable file-like object.
 *     samples: 2D array of shape [num_frames, num_channels] and np.int16 dtype.
 *     sample_rate_hz: Integer, sample rate in Hz.
 *   """
 *
 * The functions take a file object as argument (not a string filename). The
 * ReadWav*Generic() and WriteWav*Generic() C functions take callbacks for how
 * to read or write bytes to the file. We define those callbacks as wrappers
 * around Python method calls to .read() or .write() on the given Python file
 * object. This way, we can read/write WAV through any Python file object (e.g.
 * through BytesIO for an in-memory stream).
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
#include "src/dsp/read_wav_file_generic.h"
#include "src/dsp/write_wav_file_generic.h"
#include "numpy/arrayobject.h"

#define kFileBufferSize 256
typedef struct {
  char buffer[kFileBufferSize];
  PyObject* file_object;
  int remaining;
  int eof;
} FileObjectWithBuffer;

static int FillReadBuffer(FileObjectWithBuffer* f) {
  if (f->eof) { return 0; }

  /* Call the file_object's `.read(size)` method. */
  PyObject* bytes = PyObject_CallMethod(
      f->file_object, "read", "i", kFileBufferSize);
  if (!bytes) { goto fail; }

  /* Check carefully that read result is what we expect. */
  if (!PyBytes_Check(bytes)) {
    PyErr_Format(PyExc_TypeError,
                 "expected bytes, %.200s found", Py_TYPE(bytes)->tp_name);
    goto fail;
  }
  const size_t num_read = PyBytes_GET_SIZE(bytes);
  if (num_read > kFileBufferSize) {
    PyErr_SetString(PyExc_ValueError, "read result exceeds requested size");
    goto fail;
  }

  /* Save the read bytes into the buffer. */
  memcpy(f->buffer + kFileBufferSize - (int)num_read,
         PyBytes_AS_STRING(bytes), num_read);
  Py_DECREF(bytes);

  f->remaining = (int)num_read;
  f->eof = ((int)num_read < kFileBufferSize);
  return (num_read > 0); /* Return true if anything was read. */

fail:
  Py_XDECREF(bytes);
  f->remaining = 0;
  f->eof = 1;
  return 0;
}

/* Read callback for ReadWav*Generic. Reads up to `num_bytes` bytes, storing
 * them in `bytes`. Returns the number of bytes actually read, which may be less
 * on EOF or IO error.
 */
static size_t ReadFun(void* bytes, size_t num_bytes, void* io_ptr) {
  FileObjectWithBuffer* f = (FileObjectWithBuffer*)io_ptr;

  /* This function gets called often with num_bytes <= 4, so when possible take
   * a fast path that just copies `num_bytes` bytes from the buffer and returns.
   */
  if ((size_t)f->remaining >= num_bytes) {
    memcpy(bytes, f->buffer + kFileBufferSize - f->remaining, num_bytes);
    f->remaining -= (int)num_bytes;
    return num_bytes;
  }

  /* Otherwise, copy remaining bytes from the buffer, refill, and repeat. */
  size_t num_read = 0;
  do {
    /* Refill the buffer when it is empty. Break the loop on EOF or error. */
    if (!f->remaining && !FillReadBuffer(f)) { break; }

    size_t num_copy = f->remaining;
    if (num_bytes < num_copy) { num_copy = num_bytes; }
    memcpy(bytes, f->buffer + kFileBufferSize - f->remaining, num_copy);
    num_read += num_copy;
    bytes += num_copy;
    num_bytes -= num_copy;
    f->remaining -= num_copy;
  } while (num_bytes > 0);

  return num_read;
}

/* End-of-file callback for ReadWav*Generic. */
static int EndOfFileFun(void* io_ptr) {
  FileObjectWithBuffer* f = (FileObjectWithBuffer*)io_ptr;
  return f->eof && f->remaining == 0;
}

/* Converts sample format to the corresponding numpy dtype type_num. */
static int SampleFormatToNumpy(enum SampleType sample_format) {
  switch (sample_format) {
    case kInt16: return NPY_INT16;
    case kInt32: return NPY_INT32;
    case kFloat: return NPY_FLOAT32;
  }
  return NPY_NOTYPE;
}

/* Converts numpy dtype type_num to sample format. */
static enum SampleType NumpyToSampleFormat(int type_num) {
  switch (type_num) {
    case NPY_INT16: return kInt16;
    case NPY_INT32: return kInt32;
    default: return kFloat;
  }
}

/* Casts and rescales samples from src_format to dest_format.
 * Returns 1 on success, 0 on failure.
 */
static int ConvertSampleFormat(const char* src,
                               size_t num_samples,
                               enum SampleType src_format,
                               enum SampleType dest_format,
                               char* dest) {
  size_t i;
  switch (src_format) {
    case kInt16: {
        const int16_t* src_int16 = (const int16_t*)src;
        switch (dest_format) {
          case kInt16: /* int16 -> int16. */
            memcpy(dest, src, sizeof(int16_t) * num_samples);
            return 1;
          case kInt32: /* int16 -> int32. */
            for (i = 0; i < num_samples; ++i) {
              ((int32_t*)dest)[i] = ((int32_t)src_int16[i]) << 16;
            }
            return 1;
          case kFloat: /* int16 -> float32. */
            for (i = 0; i < num_samples; ++i) {
              ((float*)dest)[i] = ((float)src_int16[i]) / 32768.0f;
            }
            return 1;
        }
      }
      break;

    case kInt32: {
        const int32_t* src_int32 = (const int32_t*)src;
        switch (dest_format) {
          case kInt16: /* int32 -> int16. */
            for (i = 0; i < num_samples; ++i) {
              ((int16_t*)dest)[i] = (int16_t)(src_int32[i] / (1 << 16));
            }
            return 1;
          case kInt32: /* int32 -> int32. */
            memcpy(dest, src, sizeof(int32_t) * num_samples);
            return 1;
          case kFloat: /* int32 -> float32. */
            for (i = 0; i < num_samples; ++i) {
              ((float*)dest)[i] = ((float)src_int32[i]) / 2147483648.0f;
            }
            return 1;
        }
      }
      break;

    case kFloat: {
        const float* src_float32 = (const float*)src;
        switch (dest_format) {
          case kInt16: /* float32 -> int16. */
            for (i = 0; i < num_samples; ++i) {
              /* Scale and round. */
              float value = floor(src_float32[i] * 32768.0f + 0.5f);
              /* Replace NaNs with zero. */
              if (value != value /* nan */) { value = 0.0f; }
              /* Saturate to int16 limits. */
              if (value < -32768.0f) { value = -32768.0f; }
              if (value > 32767.0f) { value = 32767.0f; }
              ((int16_t*)dest)[i] = (int16_t)value;
            }
            return 1;
          case kInt32: /* float32 -> int32. */
            for (i = 0; i < num_samples; ++i) {
              /* NOTE: float doesn't work for saturating to int32 limits since
               * INT32_MAX = 2147483647 is not representable as a float. With
               * default rounding it becomes 2147483648.0f, which triggers UB
               * when casting int32. We circumvent this by promoting to double,
               * which thanks to its 53-bit significand can represent any int32.
               */
              double value = floor(src_float32[i] * 2147483648.0f + 0.5f);
              if (value != value /* nan */) { value = 0.0; }
              if (value < -2147483648.0) { value = -2147483648.0; }
              if (value > 2147483647.0) { value = 2147483647.0; }
              ((int32_t*)dest)[i] = (int32_t)value;
            }
            return 1;
          case kFloat: /* float32 -> float32. */
            memcpy(dest, src, sizeof(float) * num_samples);
            return 1;
        }
      }
      break;
  }

  return 0;
}

/* Implements `read_wav_impl()` Python function. */
static PyObject* ReadWavImpl(PyObject* dummy, PyObject* args, PyObject* kw) {
  PyObject* file_object_arg = NULL;
  PyArray_Descr* descr = NULL;
  static const char* keywords[] = {"file_object", "dtype", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "O|O&:read_wav_impl",
                                   (char**)keywords, &file_object_arg,
                                   PyArray_DescrConverter2, &descr)) {
    return NULL;  /* PyArg_ParseTupleAndKeywords failed. */
  } else if (!PyObject_HasAttrString(file_object_arg, "read")) {
    return PyErr_Format(PyExc_TypeError,
                        "expected file-like object, %.200s found",
                        Py_TYPE(file_object_arg)->tp_name);
  } else if (descr && descr->type_num != NPY_INT16 &&
             descr->type_num != NPY_INT32 && descr->type_num != NPY_FLOAT32) {
    return PyErr_Format(PyExc_ValueError,
                        "dtype must be one of np.int16, np.int32, np.float32");
  }

  FileObjectWithBuffer f;
  f.file_object = file_object_arg;
  f.remaining = 0;
  f.eof = 0;

  WavReader w;
  w.read_fun = ReadFun;
  w.seek_fun = NULL;
  w.eof_fun = EndOfFileFun;
  w.custom_chunk_fun = NULL;
  w.io_ptr = &f;

  /* Read WAV header. */
  ReadWavInfo info;
  if (!ReadWavHeaderGeneric(&w, &info)) {
    if (!PyErr_Occurred()) {
      PyErr_Format(PyExc_OSError, "error reading WAV header");
    }
    return NULL;
  }

  /* Allocate array for waveform sample data. */
  char* samples_data = (char*)malloc(info.remaining_samples *
                                     info.destination_alignment_bytes);
  if (samples_data == NULL) {
    return PyErr_NoMemory();
  }

  /* Read the samples. `num_samples` is the number of samples actually read. */
  const size_t num_samples = ReadWavSamplesGeneric(&w, &info, samples_data,
                                                   info.remaining_samples);
  if (PyErr_Occurred()) { goto fail; }

  npy_intp samples_dims[2];
  samples_dims[0] = num_samples / info.num_channels;
  samples_dims[1] = info.num_channels;
  PyArrayObject* samples_object = NULL;

  if (descr && descr->type_num != SampleFormatToNumpy(info.sample_format)) {
    /* Convert samples to requested dtype. */
    samples_object = (PyArrayObject*)PyArray_SimpleNew(
        2, samples_dims, descr->type_num);
    if (!samples_object) { goto fail; }

    const enum SampleType dest_format = NumpyToSampleFormat(descr->type_num);
    if (!ConvertSampleFormat(samples_data,
                             num_samples,
                             info.sample_format,
                             dest_format,
                             PyArray_DATA(samples_object))) {
      PyErr_Format(PyExc_NotImplementedError,
                   "Unhandled sample conversion (%d -> %d)",
                   info.sample_format, dest_format);
      Py_DECREF(samples_object);
      goto fail;
    }

    free(samples_data);
  } else {
    /* No dtype conversion needed. Wrap `samples_data` as numpy array. */
    samples_object = (PyArrayObject*)PyArray_SimpleNewFromData(
        2, samples_dims, SampleFormatToNumpy(info.sample_format), samples_data);
    if (!samples_object) { goto fail; }

    /* Set samples_object as the owner of samples_data. */
    PyArray_UpdateFlags(samples_object, NPY_ARRAY_OWNDATA);
  }

  return Py_BuildValue("Ni", (PyObject*)samples_object, info.sample_rate_hz);

fail:
  free(samples_data);
  return NULL;
}

/* Flushes the write buffer by calling `file_object.write(buffer)`. Returns 1 on
 * success, 0 on failure.
 */
static int FlushWriteBuffer(FileObjectWithBuffer* f) {
  const size_t count = kFileBufferSize - f->remaining;
  if (count == 0) { return 0; }

  f->remaining = kFileBufferSize;
  /* Call the file_object's `.write(s)` method. In Python 3+, the argument must
   * be a bytes object ("y#" format string). In Python 2, it must be a string
   * object ("s#").
   */
  PyObject* write_result = PyObject_CallMethod(
      f->file_object, "write", "y#", f->buffer, count);

  if (!write_result) { return 0; }
  Py_DECREF(write_result);
  return 1;
}

/* Write callback for WriteWav*Generic. */
static size_t WriteFun(
    const void* bytes, size_t num_bytes, void* io_ptr) {
  FileObjectWithBuffer* f = (FileObjectWithBuffer*)io_ptr;

  /* This function gets called often with num_bytes <= 4, so when possible take
   * a fast path that just copies `num_bytes` bytes to the buffer and returns.
   */
  if ((size_t)f->remaining >= num_bytes) {
    memcpy(f->buffer + kFileBufferSize - f->remaining, bytes, num_bytes);
    f->remaining -= (int)num_bytes;
    return num_bytes;
  }

  /* Otherwise, copy into the buffer, flush, and repeat. */
  size_t num_written = 0;
  do {
    /* Flush the buffer when it is full. Break from the loop on error. */
    if (!f->remaining && !FlushWriteBuffer(f)) { break; }

    size_t num_copy = f->remaining;
    if (num_bytes < num_copy) { num_copy = num_bytes; }
    memcpy(f->buffer + kFileBufferSize - f->remaining, bytes, num_copy);
    num_written += num_copy;
    bytes += num_copy;
    num_bytes -= num_copy;
    f->remaining -= num_copy;
  } while (num_bytes > 0);

  return num_written;
}

/* Implements `write_wav_impl()` Python function. */
static PyObject* WriteWavImpl(PyObject* dummy, PyObject* args, PyObject* kw) {
  PyObject* file_object_arg = NULL;
  PyObject* samples_arg = NULL;
  int sample_rate_hz = 0;
  static const char* keywords[] = {
    "file_object", "samples", "sample_rate_hz", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOi:write_wav_impl",
                                   (char**)keywords,
                                   &file_object_arg,
                                   &samples_arg,
                                   &sample_rate_hz)) {
    return NULL;  /* PyArg_ParseTupleAndKeywords failed. */
  } else if (!PyObject_HasAttrString(file_object_arg, "write")) {
    return PyErr_Format(PyExc_TypeError,
                        "expected writable file-like object, %.200s found",
                        Py_TYPE(file_object_arg)->tp_name);
  } else if (sample_rate_hz <= 0) {
    return PyErr_Format(PyExc_ValueError, "sample_rate_hz must be positive");
  }

  /* Convert input samples to numpy array with contiguous int16 data.
   * NOTE: We do not include NPY_ARRAY_FORCECAST in the requirements flags so
   * that casting to int16 occurs only when it can be done safely. Particularly,
   * this prevents silent data loss from passing float-valued samples.
   */
  PyArrayObject* samples = (PyArrayObject*)PyArray_FromAny(
      samples_arg, PyArray_DescrFromType(NPY_INT16), 0, 0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_DEFAULT, NULL);

  if (!samples) {  /* PyArray_DescrFromType failed. */
    return NULL;
  }

  int num_channels;
  if (PyArray_NDIM(samples) == 1) {
    num_channels = 1;  /* Interpret 1D array as a single channel. */
  } else if (PyArray_NDIM(samples) == 2) {
    num_channels = (int)PyArray_DIM(samples, 1);
  } else {
    PyErr_SetString(PyExc_ValueError, "expected 1D or 2D array");
    goto fail;
  }

  FileObjectWithBuffer f;
  f.file_object = file_object_arg;
  f.remaining = kFileBufferSize;

  WavWriter w;
  w.io_ptr = &f;
  w.write_fun = WriteFun;

  /* Write WAV header. */
  const size_t num_samples = (size_t)PyArray_SIZE(samples);
  if (!WriteWavHeaderGeneric(&w, num_samples, sample_rate_hz, num_channels)) {
    if (!PyErr_Occurred()) {
      PyErr_Format(PyExc_OSError, "error writing WAV header");
    }
    goto fail;
  }

  /* Write the samples. */
  if (!WriteWavSamplesGeneric(
      &w, (const int16_t*)PyArray_DATA(samples), num_samples) ||
      !FlushWriteBuffer(&f)) {
    if (!PyErr_Occurred()) {
      PyErr_Format(PyExc_OSError, "error writing WAV samples");
    }
    goto fail;
  }

  Py_DECREF(samples);
  Py_INCREF(Py_None);
  return Py_None;

fail:
  Py_XDECREF(samples);
  return NULL;
}

/* Module methods. */
static PyMethodDef kModuleMethods[] = {
    {"read_wav_impl", (PyCFunction)ReadWavImpl,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"write_wav_impl", (PyCFunction)WriteWavImpl,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Module definition. */
static struct PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "wav_io_python_bindings",  /* m_name */
    NULL,                        /* m_doc */
    (Py_ssize_t)-1,              /* m_size */
    kModuleMethods,              /* m_methods */
    NULL,                        /* m_reload */
    NULL,                        /* m_traverse */
    NULL,                        /* m_clear */
    NULL,                        /* m_free */
};

PyMODINIT_FUNC PyInit_wav_io_python_bindings(void) {
  import_array();
  return PyModule_Create(&kModule);
}
