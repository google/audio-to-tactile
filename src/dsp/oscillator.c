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
 */

#include "dsp/oscillator.h"

#include <math.h>

#define kPhasesPerCycle 4294967296.0f  /* = 2^32. */
#define kRoundOffset (UINT32_C(1) << (31 - kOscillatorTableBits))

void OscillatorSetFrequency(Oscillator* oscillator,
                            float frequency_cycles_per_sample) {
  frequency_cycles_per_sample -= floor(frequency_cycles_per_sample);
  oscillator->delta_phase =
      (uint32_t)(frequency_cycles_per_sample * kPhasesPerCycle);
}

void OscillatorAddFrequency(Oscillator* oscillator,
                            float frequency_cycles_per_sample) {
  frequency_cycles_per_sample -= floor(frequency_cycles_per_sample);
  oscillator->delta_phase +=
      (uint32_t)(frequency_cycles_per_sample * kPhasesPerCycle);
}

float OscillatorGetFrequency(const Oscillator* oscillator) {
  return oscillator->delta_phase / kPhasesPerCycle;
}

void OscillatorSetPhase(Oscillator* oscillator, float phase_in_cycles) {
  phase_in_cycles -= floor(phase_in_cycles);

  oscillator->phase_plus_round_offset =
      (uint32_t)(phase_in_cycles * kPhasesPerCycle) + kRoundOffset;
}

void OscillatorAddPhase(Oscillator* oscillator, float phase_in_cycles) {
  phase_in_cycles -= floor(phase_in_cycles);
  oscillator->phase_plus_round_offset +=
      (uint32_t)(phase_in_cycles * kPhasesPerCycle);
}

float OscillatorGetPhase(const Oscillator* oscillator) {
  return (oscillator->phase_plus_round_offset - kRoundOffset) / kPhasesPerCycle;
}

/* Look up table of one cycle of the sin() function. This table is also useful
 * for cos() by rotating the table by a quarter cycle,
 *
 *   sin(2 pi x) ~= kOscillatorSinTable[x / 2^TableBits],
 *   cos(2 pi x) ~= kOscillatorSinTable[(x + 0.25) / 2^TableBits].
 *
 * Generated in Python with:
 *
 *   import numpy as np
 *   N = 1024
 *   table = np.sin((2 * np.pi / N) * np.arange(N))
 *   table[[0, N//4, N//2, 3*N//4]] = [0.0, 1.0, 0.0, -1.0]
 */
const float kOscillatorSinTable[1 << kOscillatorTableBits] = {
  0.0f, 0.00613588465f, 0.0122715383f, 0.0184067299f, 0.0245412285f,
  0.0306748032f, 0.0368072229f, 0.0429382569f, 0.0490676743f, 0.0551952443f,
  0.0613207363f, 0.0674439196f, 0.0735645636f, 0.079682438f, 0.0857973123f,
  0.0919089565f, 0.0980171403f, 0.104121634f, 0.110222207f, 0.116318631f,
  0.122410675f, 0.128498111f, 0.134580709f, 0.140658239f, 0.146730474f,
  0.152797185f, 0.158858143f, 0.16491312f, 0.170961889f, 0.17700422f,
  0.183039888f, 0.189068664f, 0.195090322f, 0.201104635f, 0.207111376f,
  0.21311032f, 0.21910124f, 0.225083911f, 0.231058108f, 0.237023606f,
  0.24298018f, 0.248927606f, 0.25486566f, 0.260794118f, 0.266712757f,
  0.272621355f, 0.278519689f, 0.284407537f, 0.290284677f, 0.296150888f,
  0.302005949f, 0.30784964f, 0.31368174f, 0.319502031f, 0.325310292f,
  0.331106306f, 0.336889853f, 0.342660717f, 0.34841868f, 0.354163525f,
  0.359895037f, 0.365612998f, 0.371317194f, 0.37700741f, 0.382683432f,
  0.388345047f, 0.39399204f, 0.3996242f, 0.405241314f, 0.410843171f,
  0.41642956f, 0.422000271f, 0.427555093f, 0.433093819f, 0.438616239f,
  0.444122145f, 0.44961133f, 0.455083587f, 0.460538711f, 0.465976496f,
  0.471396737f, 0.47679923f, 0.482183772f, 0.48755016f, 0.492898192f,
  0.498227667f, 0.503538384f, 0.508830143f, 0.514102744f, 0.51935599f,
  0.524589683f, 0.529803625f, 0.53499762f, 0.540171473f, 0.545324988f,
  0.550457973f, 0.555570233f, 0.560661576f, 0.565731811f, 0.570780746f,
  0.575808191f, 0.580813958f, 0.585797857f, 0.590759702f, 0.595699304f,
  0.600616479f, 0.605511041f, 0.610382806f, 0.615231591f, 0.620057212f,
  0.624859488f, 0.629638239f, 0.634393284f, 0.639124445f, 0.643831543f,
  0.648514401f, 0.653172843f, 0.657806693f, 0.662415778f, 0.666999922f,
  0.671558955f, 0.676092704f, 0.680600998f, 0.685083668f, 0.689540545f,
  0.693971461f, 0.698376249f, 0.702754744f, 0.707106781f, 0.711432196f,
  0.715730825f, 0.720002508f, 0.724247083f, 0.72846439f, 0.732654272f,
  0.736816569f, 0.740951125f, 0.745057785f, 0.749136395f, 0.753186799f,
  0.757208847f, 0.761202385f, 0.765167266f, 0.769103338f, 0.773010453f,
  0.776888466f, 0.780737229f, 0.784556597f, 0.788346428f, 0.792106577f,
  0.795836905f, 0.799537269f, 0.803207531f, 0.806847554f, 0.810457198f,
  0.81403633f, 0.817584813f, 0.821102515f, 0.824589303f, 0.828045045f,
  0.831469612f, 0.834862875f, 0.838224706f, 0.841554977f, 0.844853565f,
  0.848120345f, 0.851355193f, 0.854557988f, 0.85772861f, 0.860866939f,
  0.863972856f, 0.867046246f, 0.870086991f, 0.873094978f, 0.876070094f,
  0.879012226f, 0.881921264f, 0.884797098f, 0.88763962f, 0.890448723f,
  0.893224301f, 0.89596625f, 0.898674466f, 0.901348847f, 0.903989293f,
  0.906595705f, 0.909167983f, 0.911706032f, 0.914209756f, 0.91667906f,
  0.919113852f, 0.921514039f, 0.923879533f, 0.926210242f, 0.92850608f,
  0.930766961f, 0.932992799f, 0.93518351f, 0.937339012f, 0.939459224f,
  0.941544065f, 0.943593458f, 0.945607325f, 0.947585591f, 0.949528181f,
  0.951435021f, 0.95330604f, 0.955141168f, 0.956940336f, 0.958703475f,
  0.960430519f, 0.962121404f, 0.963776066f, 0.965394442f, 0.966976471f,
  0.968522094f, 0.970031253f, 0.971503891f, 0.972939952f, 0.974339383f,
  0.97570213f, 0.977028143f, 0.978317371f, 0.979569766f, 0.98078528f,
  0.981963869f, 0.983105487f, 0.984210092f, 0.985277642f, 0.986308097f,
  0.987301418f, 0.988257568f, 0.98917651f, 0.99005821f, 0.990902635f,
  0.991709754f, 0.992479535f, 0.993211949f, 0.99390697f, 0.994564571f,
  0.995184727f, 0.995767414f, 0.996312612f, 0.996820299f, 0.997290457f,
  0.997723067f, 0.998118113f, 0.998475581f, 0.998795456f, 0.999077728f,
  0.999322385f, 0.999529418f, 0.999698819f, 0.999830582f, 0.999924702f,
  0.999981175f, 1.0f, 0.999981175f, 0.999924702f, 0.999830582f, 0.999698819f,
  0.999529418f, 0.999322385f, 0.999077728f, 0.998795456f, 0.998475581f,
  0.998118113f, 0.997723067f, 0.997290457f, 0.996820299f, 0.996312612f,
  0.995767414f, 0.995184727f, 0.994564571f, 0.99390697f, 0.993211949f,
  0.992479535f, 0.991709754f, 0.990902635f, 0.99005821f, 0.98917651f,
  0.988257568f, 0.987301418f, 0.986308097f, 0.985277642f, 0.984210092f,
  0.983105487f, 0.981963869f, 0.98078528f, 0.979569766f, 0.978317371f,
  0.977028143f, 0.97570213f, 0.974339383f, 0.972939952f, 0.971503891f,
  0.970031253f, 0.968522094f, 0.966976471f, 0.965394442f, 0.963776066f,
  0.962121404f, 0.960430519f, 0.958703475f, 0.956940336f, 0.955141168f,
  0.95330604f, 0.951435021f, 0.949528181f, 0.947585591f, 0.945607325f,
  0.943593458f, 0.941544065f, 0.939459224f, 0.937339012f, 0.93518351f,
  0.932992799f, 0.930766961f, 0.92850608f, 0.926210242f, 0.923879533f,
  0.921514039f, 0.919113852f, 0.91667906f, 0.914209756f, 0.911706032f,
  0.909167983f, 0.906595705f, 0.903989293f, 0.901348847f, 0.898674466f,
  0.89596625f, 0.893224301f, 0.890448723f, 0.88763962f, 0.884797098f,
  0.881921264f, 0.879012226f, 0.876070094f, 0.873094978f, 0.870086991f,
  0.867046246f, 0.863972856f, 0.860866939f, 0.85772861f, 0.854557988f,
  0.851355193f, 0.848120345f, 0.844853565f, 0.841554977f, 0.838224706f,
  0.834862875f, 0.831469612f, 0.828045045f, 0.824589303f, 0.821102515f,
  0.817584813f, 0.81403633f, 0.810457198f, 0.806847554f, 0.803207531f,
  0.799537269f, 0.795836905f, 0.792106577f, 0.788346428f, 0.784556597f,
  0.780737229f, 0.776888466f, 0.773010453f, 0.769103338f, 0.765167266f,
  0.761202385f, 0.757208847f, 0.753186799f, 0.749136395f, 0.745057785f,
  0.740951125f, 0.736816569f, 0.732654272f, 0.72846439f, 0.724247083f,
  0.720002508f, 0.715730825f, 0.711432196f, 0.707106781f, 0.702754744f,
  0.698376249f, 0.693971461f, 0.689540545f, 0.685083668f, 0.680600998f,
  0.676092704f, 0.671558955f, 0.666999922f, 0.662415778f, 0.657806693f,
  0.653172843f, 0.648514401f, 0.643831543f, 0.639124445f, 0.634393284f,
  0.629638239f, 0.624859488f, 0.620057212f, 0.615231591f, 0.610382806f,
  0.605511041f, 0.600616479f, 0.595699304f, 0.590759702f, 0.585797857f,
  0.580813958f, 0.575808191f, 0.570780746f, 0.565731811f, 0.560661576f,
  0.555570233f, 0.550457973f, 0.545324988f, 0.540171473f, 0.53499762f,
  0.529803625f, 0.524589683f, 0.51935599f, 0.514102744f, 0.508830143f,
  0.503538384f, 0.498227667f, 0.492898192f, 0.48755016f, 0.482183772f,
  0.47679923f, 0.471396737f, 0.465976496f, 0.460538711f, 0.455083587f,
  0.44961133f, 0.444122145f, 0.438616239f, 0.433093819f, 0.427555093f,
  0.422000271f, 0.41642956f, 0.410843171f, 0.405241314f, 0.3996242f,
  0.39399204f, 0.388345047f, 0.382683432f, 0.37700741f, 0.371317194f,
  0.365612998f, 0.359895037f, 0.354163525f, 0.34841868f, 0.342660717f,
  0.336889853f, 0.331106306f, 0.325310292f, 0.319502031f, 0.31368174f,
  0.30784964f, 0.302005949f, 0.296150888f, 0.290284677f, 0.284407537f,
  0.278519689f, 0.272621355f, 0.266712757f, 0.260794118f, 0.25486566f,
  0.248927606f, 0.24298018f, 0.237023606f, 0.231058108f, 0.225083911f,
  0.21910124f, 0.21311032f, 0.207111376f, 0.201104635f, 0.195090322f,
  0.189068664f, 0.183039888f, 0.17700422f, 0.170961889f, 0.16491312f,
  0.158858143f, 0.152797185f, 0.146730474f, 0.140658239f, 0.134580709f,
  0.128498111f, 0.122410675f, 0.116318631f, 0.110222207f, 0.104121634f,
  0.0980171403f, 0.0919089565f, 0.0857973123f, 0.079682438f, 0.0735645636f,
  0.0674439196f, 0.0613207363f, 0.0551952443f, 0.0490676743f, 0.0429382569f,
  0.0368072229f, 0.0306748032f, 0.0245412285f, 0.0184067299f, 0.0122715383f,
  0.00613588465f, 0.0f, -0.00613588465f, -0.0122715383f, -0.0184067299f,
  -0.0245412285f, -0.0306748032f, -0.0368072229f, -0.0429382569f,
  -0.0490676743f, -0.0551952443f, -0.0613207363f, -0.0674439196f,
  -0.0735645636f, -0.079682438f, -0.0857973123f, -0.0919089565f, -0.0980171403f,
  -0.104121634f, -0.110222207f, -0.116318631f, -0.122410675f, -0.128498111f,
  -0.134580709f, -0.140658239f, -0.146730474f, -0.152797185f, -0.158858143f,
  -0.16491312f, -0.170961889f, -0.17700422f, -0.183039888f, -0.189068664f,
  -0.195090322f, -0.201104635f, -0.207111376f, -0.21311032f, -0.21910124f,
  -0.225083911f, -0.231058108f, -0.237023606f, -0.24298018f, -0.248927606f,
  -0.25486566f, -0.260794118f, -0.266712757f, -0.272621355f, -0.278519689f,
  -0.284407537f, -0.290284677f, -0.296150888f, -0.302005949f, -0.30784964f,
  -0.31368174f, -0.319502031f, -0.325310292f, -0.331106306f, -0.336889853f,
  -0.342660717f, -0.34841868f, -0.354163525f, -0.359895037f, -0.365612998f,
  -0.371317194f, -0.37700741f, -0.382683432f, -0.388345047f, -0.39399204f,
  -0.3996242f, -0.405241314f, -0.410843171f, -0.41642956f, -0.422000271f,
  -0.427555093f, -0.433093819f, -0.438616239f, -0.444122145f, -0.44961133f,
  -0.455083587f, -0.460538711f, -0.465976496f, -0.471396737f, -0.47679923f,
  -0.482183772f, -0.48755016f, -0.492898192f, -0.498227667f, -0.503538384f,
  -0.508830143f, -0.514102744f, -0.51935599f, -0.524589683f, -0.529803625f,
  -0.53499762f, -0.540171473f, -0.545324988f, -0.550457973f, -0.555570233f,
  -0.560661576f, -0.565731811f, -0.570780746f, -0.575808191f, -0.580813958f,
  -0.585797857f, -0.590759702f, -0.595699304f, -0.600616479f, -0.605511041f,
  -0.610382806f, -0.615231591f, -0.620057212f, -0.624859488f, -0.629638239f,
  -0.634393284f, -0.639124445f, -0.643831543f, -0.648514401f, -0.653172843f,
  -0.657806693f, -0.662415778f, -0.666999922f, -0.671558955f, -0.676092704f,
  -0.680600998f, -0.685083668f, -0.689540545f, -0.693971461f, -0.698376249f,
  -0.702754744f, -0.707106781f, -0.711432196f, -0.715730825f, -0.720002508f,
  -0.724247083f, -0.72846439f, -0.732654272f, -0.736816569f, -0.740951125f,
  -0.745057785f, -0.749136395f, -0.753186799f, -0.757208847f, -0.761202385f,
  -0.765167266f, -0.769103338f, -0.773010453f, -0.776888466f, -0.780737229f,
  -0.784556597f, -0.788346428f, -0.792106577f, -0.795836905f, -0.799537269f,
  -0.803207531f, -0.806847554f, -0.810457198f, -0.81403633f, -0.817584813f,
  -0.821102515f, -0.824589303f, -0.828045045f, -0.831469612f, -0.834862875f,
  -0.838224706f, -0.841554977f, -0.844853565f, -0.848120345f, -0.851355193f,
  -0.854557988f, -0.85772861f, -0.860866939f, -0.863972856f, -0.867046246f,
  -0.870086991f, -0.873094978f, -0.876070094f, -0.879012226f, -0.881921264f,
  -0.884797098f, -0.88763962f, -0.890448723f, -0.893224301f, -0.89596625f,
  -0.898674466f, -0.901348847f, -0.903989293f, -0.906595705f, -0.909167983f,
  -0.911706032f, -0.914209756f, -0.91667906f, -0.919113852f, -0.921514039f,
  -0.923879533f, -0.926210242f, -0.92850608f, -0.930766961f, -0.932992799f,
  -0.93518351f, -0.937339012f, -0.939459224f, -0.941544065f, -0.943593458f,
  -0.945607325f, -0.947585591f, -0.949528181f, -0.951435021f, -0.95330604f,
  -0.955141168f, -0.956940336f, -0.958703475f, -0.960430519f, -0.962121404f,
  -0.963776066f, -0.965394442f, -0.966976471f, -0.968522094f, -0.970031253f,
  -0.971503891f, -0.972939952f, -0.974339383f, -0.97570213f, -0.977028143f,
  -0.978317371f, -0.979569766f, -0.98078528f, -0.981963869f, -0.983105487f,
  -0.984210092f, -0.985277642f, -0.986308097f, -0.987301418f, -0.988257568f,
  -0.98917651f, -0.99005821f, -0.990902635f, -0.991709754f, -0.992479535f,
  -0.993211949f, -0.99390697f, -0.994564571f, -0.995184727f, -0.995767414f,
  -0.996312612f, -0.996820299f, -0.997290457f, -0.997723067f, -0.998118113f,
  -0.998475581f, -0.998795456f, -0.999077728f, -0.999322385f, -0.999529418f,
  -0.999698819f, -0.999830582f, -0.999924702f, -0.999981175f, -1.0f,
  -0.999981175f, -0.999924702f, -0.999830582f, -0.999698819f, -0.999529418f,
  -0.999322385f, -0.999077728f, -0.998795456f, -0.998475581f, -0.998118113f,
  -0.997723067f, -0.997290457f, -0.996820299f, -0.996312612f, -0.995767414f,
  -0.995184727f, -0.994564571f, -0.99390697f, -0.993211949f, -0.992479535f,
  -0.991709754f, -0.990902635f, -0.99005821f, -0.98917651f, -0.988257568f,
  -0.987301418f, -0.986308097f, -0.985277642f, -0.984210092f, -0.983105487f,
  -0.981963869f, -0.98078528f, -0.979569766f, -0.978317371f, -0.977028143f,
  -0.97570213f, -0.974339383f, -0.972939952f, -0.971503891f, -0.970031253f,
  -0.968522094f, -0.966976471f, -0.965394442f, -0.963776066f, -0.962121404f,
  -0.960430519f, -0.958703475f, -0.956940336f, -0.955141168f, -0.95330604f,
  -0.951435021f, -0.949528181f, -0.947585591f, -0.945607325f, -0.943593458f,
  -0.941544065f, -0.939459224f, -0.937339012f, -0.93518351f, -0.932992799f,
  -0.930766961f, -0.92850608f, -0.926210242f, -0.923879533f, -0.921514039f,
  -0.919113852f, -0.91667906f, -0.914209756f, -0.911706032f, -0.909167983f,
  -0.906595705f, -0.903989293f, -0.901348847f, -0.898674466f, -0.89596625f,
  -0.893224301f, -0.890448723f, -0.88763962f, -0.884797098f, -0.881921264f,
  -0.879012226f, -0.876070094f, -0.873094978f, -0.870086991f, -0.867046246f,
  -0.863972856f, -0.860866939f, -0.85772861f, -0.854557988f, -0.851355193f,
  -0.848120345f, -0.844853565f, -0.841554977f, -0.838224706f, -0.834862875f,
  -0.831469612f, -0.828045045f, -0.824589303f, -0.821102515f, -0.817584813f,
  -0.81403633f, -0.810457198f, -0.806847554f, -0.803207531f, -0.799537269f,
  -0.795836905f, -0.792106577f, -0.788346428f, -0.784556597f, -0.780737229f,
  -0.776888466f, -0.773010453f, -0.769103338f, -0.765167266f, -0.761202385f,
  -0.757208847f, -0.753186799f, -0.749136395f, -0.745057785f, -0.740951125f,
  -0.736816569f, -0.732654272f, -0.72846439f, -0.724247083f, -0.720002508f,
  -0.715730825f, -0.711432196f, -0.707106781f, -0.702754744f, -0.698376249f,
  -0.693971461f, -0.689540545f, -0.685083668f, -0.680600998f, -0.676092704f,
  -0.671558955f, -0.666999922f, -0.662415778f, -0.657806693f, -0.653172843f,
  -0.648514401f, -0.643831543f, -0.639124445f, -0.634393284f, -0.629638239f,
  -0.624859488f, -0.620057212f, -0.615231591f, -0.610382806f, -0.605511041f,
  -0.600616479f, -0.595699304f, -0.590759702f, -0.585797857f, -0.580813958f,
  -0.575808191f, -0.570780746f, -0.565731811f, -0.560661576f, -0.555570233f,
  -0.550457973f, -0.545324988f, -0.540171473f, -0.53499762f, -0.529803625f,
  -0.524589683f, -0.51935599f, -0.514102744f, -0.508830143f, -0.503538384f,
  -0.498227667f, -0.492898192f, -0.48755016f, -0.482183772f, -0.47679923f,
  -0.471396737f, -0.465976496f, -0.460538711f, -0.455083587f, -0.44961133f,
  -0.444122145f, -0.438616239f, -0.433093819f, -0.427555093f, -0.422000271f,
  -0.41642956f, -0.410843171f, -0.405241314f, -0.3996242f, -0.39399204f,
  -0.388345047f, -0.382683432f, -0.37700741f, -0.371317194f, -0.365612998f,
  -0.359895037f, -0.354163525f, -0.34841868f, -0.342660717f, -0.336889853f,
  -0.331106306f, -0.325310292f, -0.319502031f, -0.31368174f, -0.30784964f,
  -0.302005949f, -0.296150888f, -0.290284677f, -0.284407537f, -0.278519689f,
  -0.272621355f, -0.266712757f, -0.260794118f, -0.25486566f, -0.248927606f,
  -0.24298018f, -0.237023606f, -0.231058108f, -0.225083911f, -0.21910124f,
  -0.21311032f, -0.207111376f, -0.201104635f, -0.195090322f, -0.189068664f,
  -0.183039888f, -0.17700422f, -0.170961889f, -0.16491312f, -0.158858143f,
  -0.152797185f, -0.146730474f, -0.140658239f, -0.134580709f, -0.128498111f,
  -0.122410675f, -0.116318631f, -0.110222207f, -0.104121634f, -0.0980171403f,
  -0.0919089565f, -0.0857973123f, -0.079682438f, -0.0735645636f, -0.0674439196f,
  -0.0613207363f, -0.0551952443f, -0.0490676743f, -0.0429382569f,
  -0.0368072229f, -0.0306748032f, -0.0245412285f, -0.0184067299f,
  -0.0122715383f, -0.00613588465f};