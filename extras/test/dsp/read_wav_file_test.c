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

#include "src/dsp/read_wav_file.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/dsp/logging.h"
#include "src/dsp/write_wav_file.h"

/* Instructions for generating arrays below:
 * Wav data can be generated using python.
import scipy.io.wavfile as wavefile
import numpy as np
wavefile.write("test16.wav", 48000, np.array([7, -2, 32767, -32768],
                                             dtype=np.int16))

Convert to 24 bit using SoX.
sox test16.wav -b 24 test24.wav

Or convert to floating point.
sox test16.wav -b 32 -e floating-point testfp.wav

Return to python and view the bits.
wavdata = open("test24.wav", 'rb').read()
print map(ord, wavdata)
*/

/* A 48kHz mono WAV file with int16_t samples {7, -2, INT16_MAX, INT16_MIN}. */
static const uint8_t kTest16BitMonoWavFile[52] = {
/*  R    I    F    F                      W    A    V   E   f    m    t    _ */
    82,  73,  70,  70,  44,  0,  0,   0,  87,  65,  86, 69, 102, 109, 116, 32,
    16,  0,   0,   0,   1,   0,  1,   0,  128, 187, 0,  0,  0,   119, 1,   0,
/*                      d    a   t    a                                      */
    2,   0,   16,  0,   100, 97, 116, 97, 8,   0,   0,  0,  7,   0,   254, 255,
    255, 127, 0,   128};

/* A 16kHz 3-channel WAV file with int16_t samples {{0, 1, 2}, {3, 4, 5}}. */
static const uint8_t kTest3ChannelWavFile[92] = {
/*  R    I    F    F                   W    A   V    E    f    m    t    _ */
    82, 73, 70, 70, 84,  0,   0,  0,   87,  65, 86,  69,  102, 109, 116, 32,
    40, 0,  0,  0,  254, 255, 3,  0,   128, 62, 0,   0,   0,   119, 1,   0,
    6,  0,  16, 0,  22,  0,   16, 0,   0,   0,  0,   0,   1,   0,   0,   0,
/*                                                        f    a    c    t   */
    0,  0,  16, 0,  128, 0,   0,  170, 0,   56, 155, 113, 102, 97,  99,  116,
/*                                     d    a   t    a                       */
    4,  0,  0,  0,  2,   0,   0,  0,   100, 97, 116, 97,  12,  0,   0,   0,
    0,  0,  1,  0,  2,   0,   3,  0,   4,   0,  5,   0};

/* A 48kHz mono WAV file with int24 samples made from
 * int16 samples {7, -2, INT16_MAX, INT16_MIN} using SoX. SoX scales up the
 * data by by 2^8, and we shift by another 8 bits to get it into the most
 * significant bits of the 32-bit container.
 */
static const uint8_t kTest24BitMonoWavFile[92] = {
/* R    I    F    F                   W    A    V    E    f    m    t    _ */
   82, 73, 70, 70, 84,  0,   0,  0,   87,  65,  86,  69,  102, 109, 116, 32,
   40, 0,  0,  0,  254, 255, 1,  0,   128, 187, 0,   0,   128, 50,  2,   0,
   3,  0,  24, 0,  22,  0,   24, 0,   4,   0,   0,   0,   1,   0,   0,   0,
/*                                                        f    a    c    t   */
   0,  0,  16, 0,  128, 0,   0,  170, 0,   56,  155, 113, 102, 97,  99,  116,
/*                                      d    a    t    a                     */
   4,  0,  0,  0,   4,   0,   0,   0,   100, 97,  116, 97,  12,  0,   0,   0,
   0,  7,  0,  0,   254, 255, 0,   255, 127, 0,   0,   128};

/* A 8kHz mono WAV file with mulaw samples, decoding to
   {29052, 20860, 18812, 31100, -7164, 716, -25980, -24956}

   Created by reading random data as a 16-bit raw audio, and letting
   sox encoded it as mulaw:

     dd bs=1 count=16 if=/dev/random \
       | sox -t raw -b 16 -e unsigned-integer  -c 1 -r 8000 - \
             -t wav -e mu-law mulaw.wav

   Followed by decoding like

     sox mulaw.wav -t wav -e signed-integer - | od -Ad -td2

   to get expected decoded samples.
*/
static const uint8_t kTestMulawWavFile[] = {
/* R   I   F   F                   W   A   V   E  f   m   t    _ */
   82, 73, 70, 70, 66,  0,  0,  0, 87, 65, 86, 69,102,109,116, 32,
   18,  0,  0,  0,  7,  0,  1,  0, 64, 31,  0,  0, 64, 31,  0,  0,
/*                        f    a   c  t                          */
    1,  0,  8,  0,  0,  0,102, 97, 99,116,  4,  0,  0,  0, 15,  0,
/*        d    a  t    a                                         */
    0,  0,100, 97,116, 97,  8,  0,  0,  0,131,139,141,129, 35,213,
    6,  7,
};

/* A 48kHz mono WAV file with float samples made from int16 samples
 * {1000, -2000, INT16_MAX, INT16_MIN} using SoX. */
static const uint8_t kTestFloatMonoWavFile[74] = {
/*  R    I    F    F                      W    A    V   E   f    m    t    _ */
    82,  73,  70,  70,  66,  0,  0,   0,  87,  65,  86, 69, 102, 109, 116, 32,
    18,  0,   0,   0,   3,   0,  1,   0,  128, 187, 0,  0,  0,   238, 2,   0,
/*                               f    a   c    t                             */
    4,   0,   32,  0,   0,   0,  102, 97, 99,  116, 4,  0,  0,   0,   4,   0,
/*            d    a    t    a                                               */
    0,   0,   100, 97,  116, 97, 16,  0,  0,   0,   0,  0,  250, 60,  0,   0,
    122, 189, 0,   254, 127, 63, 0,   0,  128, 191};

/* WAV file with a bad fact chunk. */
static const uint8_t kTestBadFactChunkWavFile[22] = {
/*  R    I    F    F                      W    A    V   E   f    a   c   t */
    82,  73,  70,  70,  95,  0,  0,   0,  87,  65,  86, 69, 102, 97, 99, 116,
/*  b   u    s    t    e    d                                              */
    98, 117, 115, 116, 101, 100};

static void WriteBytesAsFile(const char* file_name,
                             const uint8_t* bytes, size_t num_bytes) {
  FILE* f = CHECK_NOTNULL(fopen(file_name, "wb"));
  CHECK(fwrite(bytes, 1, num_bytes, f) == num_bytes);
  fclose(f);
}

static void TestReadMonoWav() {
  puts("TestReadMonoWav");
  static const int16_t kExpectedSamples[4] = {7, -2, INT16_MAX, INT16_MIN};
  const char* wav_file_name = NULL;
  int16_t* samples = NULL;
  size_t num_samples;
  int num_channels;
  int sample_rate_hz;

  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  WriteBytesAsFile(wav_file_name, kTest16BitMonoWavFile, 52);

  samples = CHECK_NOTNULL(Read16BitWavFile(wav_file_name, &num_samples,
                                           &num_channels, &sample_rate_hz));

  CHECK(num_samples == 4);
  CHECK(memcmp(samples, kExpectedSamples, sizeof(int16_t) * 4) == 0);
  CHECK(num_channels == 1);
  CHECK(sample_rate_hz == 48000);

  free(samples);
  remove(wav_file_name);
}

static void TestReadMonoWav16BitGeneric() {
  puts("TestReadMonoWav16BitGeneric");
  static const int32_t kExpectedSamples[4] = {7 << 16,
                                              -(2 << 16),
                                              2147483647,
                                              -2147483648};
  const char* wav_file_name = NULL;
  int32_t* samples = NULL;
  size_t num_samples;
  int num_channels;
  int sample_rate_hz;

  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  WriteBytesAsFile(wav_file_name, kTest16BitMonoWavFile, 52);

  samples = CHECK_NOTNULL(ReadWavFile(wav_file_name, &num_samples,
                                      &num_channels, &sample_rate_hz));

  CHECK(num_samples == 4);
  CHECK(samples[0] == kExpectedSamples[0]);
  CHECK(samples[1] == kExpectedSamples[1]);
  /* The LSBs are going to be empty since we're reading 16 bits into a
   * 32-bit container.
   */
  CHECK(samples[2] == kExpectedSamples[2] - 65535);
  CHECK(samples[3] == kExpectedSamples[3]);
  CHECK(num_channels == 1);
  CHECK(sample_rate_hz == 48000);

  free(samples);
  remove(wav_file_name);
}

static void TestReadMonoWav24BitGeneric() {
  puts("TestReadMonoWav24BitGeneric");
  static const int32_t kExpectedSamples[4] = {
      7 << 16,
      -(2 << 16),
      2147418112,  /* = INT16_MAX << 16. */
      -2147483648}; /* = INT16_MIN << 16. */
  const char* wav_file_name = NULL;
  int32_t* samples = NULL;
  size_t num_samples;
  int num_channels;
  int sample_rate_hz;

  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  WriteBytesAsFile(wav_file_name, kTest24BitMonoWavFile, 92);

  samples = CHECK_NOTNULL(ReadWavFile(wav_file_name, &num_samples,
                                      &num_channels, &sample_rate_hz));
  CHECK(num_samples == 4);
  CHECK(memcmp(samples, kExpectedSamples, sizeof(kExpectedSamples)) == 0);
  CHECK(num_channels == 1);
  CHECK(sample_rate_hz == 48000);

  free(samples);
  remove(wav_file_name);
}

static void TestReadMonoWavFloatGeneric() {
  puts("TestReadMonoWavFloatGeneric");
  /* The LSBs are going to be empty since we're reading 16 bits into a 32-bit
   * container (third element). */
  static const int32_t kExpectedSamples[4] = {1000 << 16,
                                              -(2000 << 16),
                                              2147483647 - 65535,
                                              -2147483648};
  const char* wav_file_name = NULL;
  int32_t* samples = NULL;
  size_t num_samples;
  int num_channels;
  int sample_rate_hz;

  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  WriteBytesAsFile(wav_file_name, kTestFloatMonoWavFile, 74);

  samples = CHECK_NOTNULL(ReadWavFile(wav_file_name, &num_samples,
                                      &num_channels, &sample_rate_hz));
  CHECK(num_samples == 4);
  CHECK(memcmp(samples, kExpectedSamples, sizeof(kExpectedSamples)) == 0);
  CHECK(num_channels == 1);
  CHECK(sample_rate_hz == 48000);

  free(samples);
  remove(wav_file_name);
}

static void TestReadMonoWavStreaming() {
  puts("TestReadMonoWavStreaming");
  const char* wav_file_name = NULL;
  FILE* f;
  ReadWavInfo info;
  int16_t buffer[3];

  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  WriteBytesAsFile(wav_file_name, kTest16BitMonoWavFile, 52);

  f = CHECK_NOTNULL(fopen(wav_file_name, "rb"));
  CHECK(ReadWavHeader(f, &info));
  CHECK(info.num_channels == 1);
  CHECK(info.sample_rate_hz == 48000);
  CHECK(info.remaining_samples == 4);

  CHECK(Read16BitWavSamples(f, &info, buffer, 3) == 3);
  CHECK(buffer[0] == 7);
  CHECK(buffer[1] == -2);
  CHECK(buffer[2] == INT16_MAX);
  CHECK(Read16BitWavSamples(f, &info, buffer, 3) == 1);
  CHECK(buffer[0] == INT16_MIN);

  fclose(f);
  remove(wav_file_name);
}

static void TestRead3ChannelWav() {
  puts("TestRead3ChannelWav");
  static const int16_t kExpectedSamples[6] = {0, 1, 2, 3, 4, 5};
  const char* wav_file_name = NULL;
  int16_t* samples = NULL;
  size_t num_samples;
  int num_channels;
  int sample_rate_hz;

  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  WriteBytesAsFile(wav_file_name, kTest3ChannelWavFile, 92);

  samples = CHECK_NOTNULL(Read16BitWavFile(wav_file_name, &num_samples,
                                           &num_channels, &sample_rate_hz));
  CHECK(num_samples == 6);
  CHECK(memcmp(samples, kExpectedSamples, sizeof(int16_t) * 6) == 0);
  CHECK(num_channels == 3);
  CHECK(sample_rate_hz == 16000);

  free(samples);
  remove(wav_file_name);
}

static void TestReadMulawWav() {
  puts("TestReadMulawWav");
  static const int16_t kExpectedSamples[8] = {
    29052, 20860, 18812, 31100, -7164, 716, -25980, -24956
  };
  const char* wav_file_name = NULL;
  int16_t* samples = NULL;
  size_t num_samples;
  int num_channels;
  int sample_rate_hz;

  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  WriteBytesAsFile(wav_file_name, kTestMulawWavFile, sizeof(kTestMulawWavFile));

  samples = CHECK_NOTNULL(Read16BitWavFile(wav_file_name, &num_samples,
                                           &num_channels, &sample_rate_hz));
  CHECK(num_samples == 8);
  CHECK(memcmp(samples, kExpectedSamples, sizeof(int16_t) * 8) == 0);
  CHECK(num_channels == 1);
  CHECK(sample_rate_hz == 8000);

  free(samples);
  remove(wav_file_name);
}

static void TestReadMulawWavGeneric() {
  puts("TestReadMulawWavGeneric");
  static const int16_t kExpected16BitSamples[8] = {
    29052, 20860, 18812, 31100, -7164, 716, -25980, -24956
  };
  int32_t expected_samples[8];
  const char* wav_file_name = NULL;
  int32_t* samples = NULL;
  size_t num_samples;
  int num_channels;
  int sample_rate_hz;
  int i;

  for (i = 0; i < 8; i++) {
    expected_samples[i] = (int32_t) kExpected16BitSamples[i] << 16;
  }
  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  WriteBytesAsFile(wav_file_name, kTestMulawWavFile, sizeof(kTestMulawWavFile));

  samples = CHECK_NOTNULL(ReadWavFile(wav_file_name, &num_samples,
                                      &num_channels, &sample_rate_hz));
  CHECK(num_samples == 8);
  CHECK(memcmp(samples, expected_samples, sizeof(int32_t) * 8) == 0);
  CHECK(num_channels == 1);
  CHECK(sample_rate_hz == 8000);

  free(samples);
  remove(wav_file_name);
}

static void TestReadBadWavTruncatedFile() {
  puts("TestReadBadWavTruncatedFile");
  const char* wav_file_name = NULL;
  int16_t* samples = NULL;
  size_t num_samples;
  int num_channels;
  int sample_rate_hz;

  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  WriteBytesAsFile(wav_file_name, kTest3ChannelWavFile, 74 /* cut 18 bytes */);

  /* Reading should fail. */
  CHECK(Read16BitWavFile(wav_file_name, &num_samples,
                         &num_channels, &sample_rate_hz) == NULL);
  /* Check that obtained WAV data is empty. */
  CHECK(samples == NULL);
  CHECK(num_samples == 0);
  CHECK(num_channels == 0);
  CHECK(sample_rate_hz == 0);
  remove(wav_file_name);
}

static void TestReadBadFactChunk() {
  puts("TestReadBadFactChunk");
  const char* wav_file_name = NULL;
  int16_t* samples = NULL;
  size_t num_samples;
  int num_channels;
  int sample_rate_hz;

  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));
  WriteBytesAsFile(wav_file_name, kTestBadFactChunkWavFile, 22);

  /* Reading should fail. */
  CHECK(Read16BitWavFile(wav_file_name, &num_samples,
                         &num_channels, &sample_rate_hz) == NULL);
  /* Check that obtained WAV data is empty. */
  CHECK(samples == NULL);
  CHECK(num_samples == 0);
  CHECK(num_channels == 0);
  CHECK(sample_rate_hz == 0);
  remove(wav_file_name);
}

static void TestWriteReadRoundTrips() {
  puts("TestWriteReadRoundTrips");
  const char* wav_file_name = NULL;
  int num_channels;

  wav_file_name = CHECK_NOTNULL(tmpnam(NULL));

  for (num_channels = 1; num_channels <= 8; ++num_channels) {
    int sample_rate_hz = rand() % 100000;
    size_t num_samples = 32 + (rand() % 100);
    int16_t* samples = CHECK_NOTNULL((int16_t*) malloc(
        sizeof(int16_t) * num_samples));
    int16_t* read_samples = NULL;
    size_t read_num_samples;
    int read_num_channels;
    int read_sample_rate_hz;
    int i;

    /* Write a WAV file with random samples. */
    num_samples -= num_samples % num_channels;
    for (i = 0; i < num_samples; ++i) {
      samples[i] = (rand() % 2001) - 1000;
    }
    CHECK(WriteWavFile(wav_file_name, samples, num_samples,
                       sample_rate_hz, num_channels));

    /* Read the WAV and check that the data was recovered successfully. */
    read_samples = CHECK_NOTNULL(Read16BitWavFile(wav_file_name,
        &read_num_samples, &read_num_channels, &read_sample_rate_hz));
    CHECK(read_num_samples == num_samples);
    CHECK(memcmp(read_samples, samples, sizeof(int16_t) * num_samples) == 0);
    CHECK(read_num_channels == num_channels);
    CHECK(read_sample_rate_hz == sample_rate_hz);

    free(read_samples);
    free(samples);
  }

  remove(wav_file_name);
}

int main(int argc, char** argv) {
  srand(0);
  TestReadMonoWav();
  TestReadMonoWav16BitGeneric();
  TestReadMonoWav24BitGeneric();
  TestReadMonoWavFloatGeneric();
  TestReadMonoWavStreaming();
  TestRead3ChannelWav();
  TestReadMulawWav();
  TestReadMulawWavGeneric();
  TestReadBadWavTruncatedFile();
  TestReadBadFactChunk();
  TestWriteReadRoundTrips();

  puts("PASS");
  return EXIT_SUCCESS;
}
