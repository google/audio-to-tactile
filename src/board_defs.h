// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
// Definitions that are shared between the libraries.
// One correct board only has to be selected: puck or sleeve.

#ifndef AUDIO_TO_TACTILE_SRC_BOARD_DEFS_H_
#define AUDIO_TO_TACTILE_SRC_BOARD_DEFS_H_

// Define which board is programmed.
#define SLEEVE_BOARD 1
#define PUCK_BOARD 0
#define SLIM_BOARD 0

#if SLEEVE_BOARD
#define kLedPinBlue 12
#define kThermistorPin 3
#endif

#if PUCK_BOARD
#define kLedPinBlue 19
#endif

#if SLIM_BOARD
#define kLedPinBlue 45
#define kLedPinGreen 36
#define kThermistorPin 3
#endif

#endif  // AUDIO_TO_TACTILE_SRC_BOARD_DEFS_H_
