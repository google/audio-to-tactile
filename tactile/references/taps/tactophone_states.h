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

#ifndef AUDIO_TO_TACTILE_TACTILE_REFERENCES_TAPS_TACTOPHONE_STATES_H_
#define AUDIO_TO_TACTILE_TACTILE_REFERENCES_TAPS_TACTOPHONE_STATES_H_

#include "tactile/references/taps/tactophone_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The main menu screen. Program starts here. */
extern const TactophoneState kTactophoneStateMainMenu;

/* Phoneme "free play" screen. */
extern const TactophoneState kTactophoneStateFreePlay;

/* "Test tactors" screen. */
extern const TactophoneState kTactophoneStateTestTactors;

/* States for lessons are
 *
 *   BeginLesson     Shows instructions.
 *   LessonTrial     Plays a pattern, asks what it is.
 *   LessonReview    Shows answer, allows comparison.
 *   LessonDone      Shows accuracy and other stats.
 *
 * The lesson begins in BeginLesson. The state then alternates between
 * LessonTrial and LessonCompare, until enough trials have been
 * completed. The lesson ends in LessonDone.
 */
extern const TactophoneState kTactophoneStateBeginLesson;
extern const TactophoneState kTactophoneStateLessonTrial;
extern const TactophoneState kTactophoneStateLessonReview;
extern const TactophoneState kTactophoneStateLessonDone;

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AUDIO_TO_TACTILE_TACTILE_REFERENCES_TAPS_TACTOPHONE_STATES_H_ */
