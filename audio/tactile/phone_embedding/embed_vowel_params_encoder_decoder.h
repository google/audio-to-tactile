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
 *
 *
 * Trained parameters for vowel network.
 */

#ifndef AUDIO_TACTILE_PHONE_EMBEDDING_EMBED_VOWEL_PARAMS_H_
#define AUDIO_TACTILE_PHONE_EMBEDDING_EMBED_VOWEL_PARAMS_H_

/* Number of frames in the network's input. */
#define kNumFrames 3
/* Number of CARL channels in a frame. */
#define kNumChannels 56
/* Size of an embedded frame. */
#define kEmbeddedFrameSize 3

/* Size of the decoder input. */
#define kDecoderInputSize (kNumFrames * kEmbeddedFrameSize)

/* Number of units in the fully connected layers. */
#define kEncoderDense1Units 16
#define kEncoderDense2Units 16
#define kEncoderDense3Units 16
#define kEncoderDense4Units 2
#define kDecoderDense1Units 16

/* Number of vowel targets. */
#define kNumTargets 8

static const float kEncoderDense1Weights[kNumChannels * kEncoderDense1Units] = {};

static const float kEncoderDense1Bias[kEncoderDense1Units] = {};

static const float kEncoderDense2Weights[
  kEncoderDense1Units * kEncoderDense2Units] = {};

static const float kEncoderDense2Bias[kEncoderDense2Units] = {};

static const float kEncoderDense3Weights[
  kEncoderDense2Units * kEncoderDense3Units] = {};

static const float kEncoderDense3Bias[kEncoderDense3Units] = {};

static const float kEncoderDense4Weights[
  kEncoderDense3Units * kEncoderDense4Units] = {};

static const float kEncoderDense4Bias[kEncoderDense4Units] = {};

static const float kDecoderDense1Weights[
  kDecoderInputSize * kDecoderDense1Units] = {};

static const float kDecoderDense1Bias[kDecoderDense1Units] = {};

static const float kDecoderDense2Weights[
  kDecoderDense1Units * kNumTargets] = {};

static const float kDecoderDense2Bias[kNumTargets] = {};

#endif /* AUDIO_TACTILE_PHONE_EMBEDDING_EMBED_VOWEL_PARAMS_H_ */

