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
 */

#include "src/dsp/read_wav_file_generic.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/dsp/logging.h"

/* A 48kHz mono WAV file with int16_t samples {7, -2, INT16_MAX, INT16_MIN}.
 * See read_wav_file_test for more details about how this was generated.  */
static const uint8_t kTest16BitMonoWavFile[52] = {
/*  R    I    F    F                      W    A    V   E   f    m    t    _ */
    82,  73,  70,  70,  44,  0,  0,   0,  87,  65,  86, 69, 102, 109, 116, 32,
    16,  0,   0,   0,   1,   0,  1,   0,  128, 187, 0,  0,  0,   119, 1,   0,
/*                      d    a   t    a                                      */
    2,   0,   16,  0,   100, 97, 116, 97, 8,   0,   0,  0,  7,   0,   254, 255,
    255, 127, 0,   128};

/* A WAV file encoded with a nonstandard chunk denoted by tag 'cust'. */
static const uint8_t kNonStandardWavFile[56] = {
/*  R    I   F   F                        W    A    V   E   f    m    t    _ */
    82,  73, 70, 70,  48,  0,   0,   0,   87,  65,  86, 69, 102, 109, 116, 32,
    16,  0,  0,  0,   1,   0,   1,   0,   128, 187, 0,  0,  0,   119, 1,   0,
/*                    c    u    s    t                      a    d    p    r */
    2,   0,  16, 0,   99,  117, 115, 116, 4,   0,   0,  0,  97,  100, 112, 114,
/*  d    a    t    a                                     */
    100, 97,  116, 97,  0,   0,   0,   0};

struct TestData {
  uint8_t /* bool */ found_custom_chunk;
  char custom_chunk_contents[4];  /* We know the chunk is of size 4. */
  int chunk_size;
  FILE* f;
};

typedef struct TestData TestData;

TestData MakeTestData(FILE* file) {
  TestData data;
  data.found_custom_chunk = 0;
  memset(data.custom_chunk_contents, 0, 4);
  data.chunk_size = 0;
  data.f = file;
  return data;
}

/* An implementation of standard I/O callbacks. */
static int Seek(size_t num_bytes, void* io_ptr) {
  return fseek(((TestData*)(io_ptr))->f, num_bytes, SEEK_CUR);
}

static int EndOfFile(void* io_ptr) {
  return feof(((TestData*)(io_ptr))->f);
}

static size_t ReadBytes(void* bytes, size_t num_bytes, void* io_ptr) {
  return fread(bytes, 1, num_bytes, ((TestData*)(io_ptr))->f);
}

static void HandleCustomChunk(
    char (*id)[4], const void * data, size_t num_bytes, void* io_ptr) {
  TestData* result = (TestData*)(io_ptr);
  result->found_custom_chunk = 1;
  if (memcmp(*id, "cust", 4) == 0) {
    if (num_bytes <= 4) {
      memcpy(result->custom_chunk_contents, data, num_bytes);
    }
    result->chunk_size = num_bytes;
  }
}

static WavReader CustomChunkWavReader(TestData* data) {
  WavReader w;
  w.read_fun = ReadBytes;
  w.seek_fun = Seek;
  w.eof_fun = EndOfFile;
  w.custom_chunk_fun = HandleCustomChunk;
  w.io_ptr = data;
  return w;
}

static void WriteBytesAsFile(const char* file_name,
                             const uint8_t* bytes, size_t num_bytes) {
  FILE* f = CHECK_NOTNULL(fopen(file_name, "wb"));
  CHECK(fwrite(bytes, 1, num_bytes, f) == num_bytes);
  fclose(f);
}

void TestReadMonoWav() {
  const char* wav_file_name = NULL;
  FILE* f;
  ReadWavInfo info;

  puts("Running TestReadMonoWavStreaming");
  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  WriteBytesAsFile(wav_file_name, kTest16BitMonoWavFile, 52);

  f = CHECK_NOTNULL(fopen(wav_file_name, "rb"));

  TestData data = MakeTestData(f);
  WavReader reader = CustomChunkWavReader(&data);
  CHECK(ReadWavHeaderGeneric(&reader, &info));
  CHECK(info.num_channels == 1);
  CHECK(info.sample_rate_hz == 48000);
  CHECK(info.remaining_samples == 4);

  CHECK(!data.found_custom_chunk);

  fclose(f);
}

void TestNonstandardWavFile() {
  const char* wav_file_name = NULL;
  FILE* f;
  ReadWavInfo info;

  puts("Running TestNonstandardWavFile");
  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  WriteBytesAsFile(wav_file_name, kNonStandardWavFile, 56);

  f = CHECK_NOTNULL(fopen(wav_file_name, "rb"));

  TestData data = MakeTestData(f);
  WavReader reader = CustomChunkWavReader(&data);
  CHECK(ReadWavHeaderGeneric(&reader, &info));
  CHECK(info.num_channels == 1);
  CHECK(info.sample_rate_hz == 48000);
  CHECK(info.remaining_samples == 0);
  CHECK(data.found_custom_chunk);
  CHECK(memcmp(data.custom_chunk_contents, "adpr", 4) == 0);
  CHECK(data.chunk_size == 4);

  fclose(f);
}

int main(int argc, char** argv) {
  srand(0);
  TestReadMonoWav();
  TestNonstandardWavFile();

  puts("PASS");
  return EXIT_SUCCESS;
}
