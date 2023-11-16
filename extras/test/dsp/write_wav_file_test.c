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
 */

#include "src/dsp/write_wav_file.h"

#include <string.h>

#include "src/dsp/logging.h"
#include "src/dsp/write_wav_file_generic.h"

/*
 * Instructions for generating arrays below:
 * Wav data can be generated using python.
 * import scipy.io.wavfile as wavefile
 * import numpy as np
 * wavefile.write("test16.wav", 48000, np.array([7, -2, 32767, -32768],
 *                                              dtype=np.int16))
 *
 * Convert to 24 bit using SoX.
 * sox test16.wav -b 24 test24.wav
 *
 * Convert to 32 bit using SoX.
 * sox test16.wav -b 32 test32.wav
 *
 * Or convert to floating point.
 * sox test16.wav -b 32 -e floating-point testfp.wav
 *
 * Return to python and view the bits.
 * wavdata = open("test24.wav", 'rb').read()
 * print(map(ord, wavdata))
 *
 * or alternatively, view them using xxd:
 * xxd -i this_is_a_test.wav
 */
/* A 48kHz mono WAV file with int16_t samples {7, -2, INT16_MAX, INT16_MIN}. */
static const uint8_t kTestMonoWavFile[52] = {
    82,  73,  70, 70,  44, 0, 0, 0,   87,  65,  86,  69,  102,
    109, 116, 32, 16,  0,  0, 0, 1,   0,   1,   0,   128, 187,
    0,   0,   0,  119, 1,  0, 2, 0,   16,  0,   100, 97,  116,
    97,  8,   0,  0,   0,  7, 0, 254, 255, 255, 127, 0,   128};
/* A 16kHz 3-channel WAV file with int16_t samples {{0, 1, 2}, {3, 4, 5}}. */
static const uint8_t kTest3ChannelWavFile[92] = {
    82, 73, 70, 70, 84,  0,   0,  0,   87,  65, 86,  69,  102, 109, 116, 32,
    40, 0,  0,  0,  254, 255, 3,  0,   128, 62, 0,   0,   0,   119, 1,   0,
    6,  0,  16, 0,  22,  0,   16, 0,   0,   0,  0,   0,   1,   0,   0,   0,
    0,  0,  16, 0,  128, 0,   0,  170, 0,   56, 155, 113, 102, 97,  99,  116,
    4,  0,  0,  0,  2,   0,   0,  0,   100, 97, 116, 97,  12,  0,   0,   0,
    0,  0,  1,  0,  2,   0,   3,  0,   4,   0,  5,   0};

/* A 48kHz mono WAV file with int24 samples made from
 * int16 samples {7, -2, INT16_MAX, INT16_MIN} using SoX. SoX scales up the
 * data up by 2^8, and we shift by another 8 bits to get it into the most
 * significant bits of the 32-bit container.
 */
static const uint8_t kTest24BitMonoWavFile[92] = {
/*  R   I   F   F                      W    A    V    E    f    m    t    _  */
    82, 73, 70, 70, 84,  0,   0,  0,   87,  65,  86,  69,  102, 109, 116, 32,
    40, 0,  0,  0,  254, 255, 1,  0,   128, 187, 0,   0,   128, 50,  2,   0,
    3,  0,  24, 0,  22,  0,   24, 0,   4,   0,   0,   0,   1,   0,   0,   0,
/*                                                         f    a    c    t  */
    0,  0,  16, 0,  128, 0,   0,  170, 0,   56,  155, 113, 102, 97,  99,  116,
/*                                     d    a    t    a  */
    4,  0,  0,  0,  4,   0,   0,  0,   100, 97,  116, 97,  12,  0,   0,   0,
    0,  7,  0,  0,  254, 255, 0,  255, 127, 0,   0,   128};

/* A 48kHz mono WAV file with int32 samples made from
 * int16 samples {7, -2, INT16_MAX, INT16_MIN} using SoX. SoX scales up the
 * data by by 2^16.
 */
static const uint8_t kTest32BitMonoWavFile[96] = {
/*  R   I   F   F                       W    A    V    E    f    m    t    _ */
    82, 73, 70, 70, 88,  0,   0,   0,   87,  65,  86,  69,  102, 109, 116, 32,
    40, 0,  0,  0,  254, 255, 1,   0,   128, 187, 0,   0,   0,   238,  2,  0,
    4,  0,  32, 0,  22,  0,   32,  0,   4,   0,   0,   0,   1,   0,    0,  0,
/*                                                          f    a    c    t */
    0,  0,  16, 0,  128, 0,   0,   170, 0,   56,  155, 113, 102, 97,  99,  116,
/*                                      d    a    t    a                     */
    4,  0,  0,  0,  4,   0,   0,   0,   100, 97,  116, 97,  16,  0,   0,   0,
    0,  0,  7,  0,  0,   0,   254, 255, 0,   0,   255, 127, 0,   0,   0,   128};

/* A 48kHz mono WAV file with int24 samples made from
 * int16 samples {7, -2, INT16_MAX}.
 */
static const uint8_t kTest24BitMonoOddWavFile[90] = {
/*  R   I   F   F                      W    A    V    E    f    m    t    _  */
    82, 73, 70, 70, 82,  0,   0,  0,   87,  65,  86,  69,  102, 109, 116, 32,
    40, 0,  0,  0,  254, 255, 1,  0,   128, 187, 0,   0,   128, 50,  2,   0,
    3,  0,  24, 0,  22,  0,   24, 0,   4,   0,   0,   0,   1,   0,   0,   0,
/*                                                         f    a    c    t  */
    0,  0,  16, 0,  128, 0,   0,  170, 0,   56,  155, 113, 102, 97,  99,  116,
/*                                     d    a    t    a  */
    4,  0,  0,  0,  3,   0,   0,  0,   100, 97,  116, 97,  9,   0,   0,   0,
/*                                          pad */
    0,  7,  0,  0,  254, 255, 0,  255, 127, 0};

static void CheckFileBytes(const char* file_name, const uint8_t* expected_bytes,
                           size_t num_bytes) {
  uint8_t* bytes = CHECK_NOTNULL((uint8_t *) malloc(num_bytes + 1));
  FILE* f = CHECK_NOTNULL(fopen(file_name, "rb"));
  CHECK(fread(bytes, 1, num_bytes + 1, f) == num_bytes);
  fclose(f);
  CHECK(memcmp(bytes, expected_bytes, num_bytes) == 0);
  free(bytes);
}

static void TestWriteMonoWav(void) {
  puts("TestWriteMonoWav");
  static const int16_t kSamples[4] = {7, -2, INT16_MAX, INT16_MIN};
  const char* wav_file_name = NULL;

  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  CHECK(WriteWavFile(wav_file_name, kSamples, 4, 48000, 1));

  CheckFileBytes(wav_file_name, kTestMonoWavFile, 52);
  remove(wav_file_name);
}

static void TestWriteMonoWavStreaming(void) {
  puts("TestWriteMonoWavStreaming");
  static const int16_t kSamples[4] = {7, -2, INT16_MAX, INT16_MIN};
  const char* wav_file_name = NULL;

  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  FILE* f = NULL;
  CHECK(f = fopen(wav_file_name, "wb"));
  CHECK(WriteWavHeader(f, 0, 48000, 1)); /* Write a dummy header. */
  CHECK(WriteWavSamples(f, kSamples + 0, 2));
  CHECK(WriteWavSamples(f, kSamples + 2, 2));
  fseek(f, 0, SEEK_SET);
  CHECK(WriteWavHeader(f, 4, 48000, 1));
  fclose(f);

  CheckFileBytes(wav_file_name, kTestMonoWavFile, 52);
  remove(wav_file_name);
}

static void TestWrite3ChannelWav(void) {
  puts("TestWrite3ChannelWav");
  static const int16_t kSamples[6] = {0, 1, 2, 3, 4, 5};
  const char* wav_file_name = NULL;

  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  CHECK(WriteWavFile(wav_file_name, kSamples, 6, 16000, 3));
  CheckFileBytes(wav_file_name, kTest3ChannelWavFile, 92);
  remove(wav_file_name);
}

static void TestWriteMono24BitWav(void) {
  puts("TestWriteMono24BitWav");
  /* << 8 accounts for the 16 to 24 bit conversion as described in the process
   * for generating the WAV file above. The remaining << 8 accounts for our
   * convention that the full range of int32 is utilized, not just the
   * lowest 24 bits. */
  static const int32_t kSamples[4] = {7 << 16,
                                      -(2 << 16),
                                      INT16_MAX << 16,
                                      INT16_MIN * (1 << 16)};
  const char* wav_file_name = NULL;

  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  CHECK(WriteWavFile24Bit(wav_file_name, kSamples, 4, 48000, 1));

  CheckFileBytes(wav_file_name, kTest24BitMonoWavFile, 92);
  remove(wav_file_name);
}

static void TestWriteMono24BitWavPadding(void) {
  puts("TestWriteMono24BitWavPadding");
  /* An odd number of samples such that the samples do not align to 16-bits. A
   * pad byte should get appended to the samples. */
  static const int32_t kSamples[3] = {7 << 16,
                                      -(2 << 16),
                                      INT16_MAX << 16};
  const char* wav_file_name = NULL;

  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  CHECK(WriteWavFile24Bit(wav_file_name, kSamples, 3, 48000, 1));

  CheckFileBytes(wav_file_name, kTest24BitMonoOddWavFile, 90);
  remove(wav_file_name);
}

static void TestWriteMono32BitWav(void) {
  puts("TestWriteMono32BitWav");
  /* << 16 accounts for the 16 to 32 bit conversion as described in the process
   * for generating the WAV file above. */
  static const int32_t kSamples[4] = {7 << 16,
                                      -(2 << 16),
                                      INT16_MAX << 16,
                                      INT16_MIN * (1 << 16)};
  const char* wav_file_name = NULL;

  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  CHECK(WriteWavFile32Bit(wav_file_name, kSamples, 4, 48000, 1));

  CheckFileBytes(wav_file_name, kTest32BitMonoWavFile, 96);
  remove(wav_file_name);
}

int main(int argc, char** argv) {
  TestWriteMonoWav();
  TestWriteMonoWavStreaming();
  TestWrite3ChannelWav();
  TestWriteMono24BitWav();
  TestWriteMono24BitWavPadding();
  TestWriteMono32BitWav();

  puts("PASS");
  return EXIT_SUCCESS;
}
