# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

$(warning DEPRECATION WARNING - This makefile is no longer up to date! Use bazel to build instead.)

CFLAGS += -O3 -g -I. -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-sign-compare
LDFLAGS += -lm

# Compile flags for Python bindings. Set Python version for your installation.
PYTHON_BINDINGS_CFLAGS = \
		$$(pkg-config --cflags python-3.6m) -Wno-missing-field-initializers -Wno-cast-function-type $(CFLAGS)

PROGRAMS=run_tactile_processor energy_envelope.so tactile_processor.so tactile_worker.so tactophone tactometer play_buzz run_energy_envelope run_energy_envelope_on_wav run_auto_gain_control_on_wav run_yuan2005 run_bratakos2001
BENCHMARKS=tactile_processor_benchmark
TESTS=auto_gain_control_test butterworth_test complex_test elliptic_fun_test fast_fun_test math_constants_test phasor_rotator_test read_wav_file_test read_wav_file_generic_test serialize_test write_wav_file_test carl_frontend_test embed_vowel_test phoneme_code_test tactophone_engine_test tactophone_lesson_test energy_envelope_test channel_map_test hexagon_interpolation_test nn_ops_test tactile_player_test tactile_processor_test util_test yuan2005_test

RUN_TACTILE_PROCESSOR_OBJS=audio/tactile/run_tactile_processor.o audio/tactile/run_tactile_processor_assets.o audio/dsp/portable/number_util.o audio/dsp/portable/read_wav_file.o audio/dsp/portable/read_wav_file_generic.o audio/tactile/channel_map.o audio/tactile/portaudio_device.o audio/tactile/util.o audio/tactile/sdl/basic_sdl_app.o audio/tactile/sdl/texture_from_rle_data.o audio/tactile/sdl/window_icon.o audio/tactile/tactile_processor.a

ENERGY_ENVELOPE_PYTHON_BINDINGS_OBJS=audio/tactile/python/energy_envelope_python_bindings.PICo audio/tactile/energy_envelope/energy_envelope.PICo audio/dsp/portable/butterworth.PICo audio/dsp/portable/complex.PICo audio/dsp/portable/fast_fun.PICo

TACTILE_PROCESSOR_PYTHON_BINDINGS_OBJS=audio/tactile/python/tactile_processor_python_bindings.PICo audio/tactile/tactile_processor.PICa

TACTILE_WORKER_PYTHON_BINDINGS_OBJS=audio/tactile/python/tactile_worker_python_bindings.PICo audio/tactile/python/tactile_worker.PICo audio/tactile/tactile_processor.PICa audio/tactile/channel_map.PICo audio/tactile/portaudio_device.PICo audio/tactile/util.PICo

TACTOPHONE_OBJS=audio/tactile/references/taps/tactophone_main.o audio/tactile/references/taps/tactophone_state_main_menu.o audio/tactile/references/taps/tactophone_state_free_play.o audio/tactile/references/taps/tactophone_state_test_tactors.o audio/tactile/references/taps/tactophone_state_begin_lesson.o audio/tactile/references/taps/tactophone_state_lesson_trial.o audio/tactile/references/taps/tactophone_state_lesson_review.o audio/tactile/references/taps/tactophone_state_lesson_done.o audio/tactile/references/taps/phoneme_code.o audio/tactile/references/taps/tactophone_engine.o audio/tactile/references/taps/tactophone.o audio/tactile/references/taps/tactophone_lesson.o audio/tactile/util.o audio/tactile/references/taps/tactile_player.o audio/tactile/channel_map.o

TACTOMETER_OBJS=audio/tactile/tactometer.o audio/tactile/portaudio_device.o audio/tactile/sdl/basic_sdl_app.o audio/tactile/sdl/draw_text.o audio/tactile/sdl/window_icon.o audio/tactile/util.o

PLAY_BUZZ_OBJS=audio/tactile/play_buzz.o audio/tactile/util.o

RUN_ENERGY_ENVELOPE_OBJS=audio/tactile/energy_envelope/run_energy_envelope.o audio/dsp/portable/butterworth.o audio/dsp/portable/complex.o audio/dsp/portable/fast_fun.o audio/tactile/portaudio_device.o audio/tactile/util.o audio/tactile/energy_envelope/energy_envelope.o

RUN_ENERGY_ENVELOPE_ON_WAV_OBJS=audio/tactile/energy_envelope/run_energy_envelope_on_wav.o audio/dsp/portable/butterworth.o audio/dsp/portable/complex.o audio/dsp/portable/fast_fun.o audio/dsp/portable/read_wav_file.o audio/dsp/portable/read_wav_file_generic.o audio/dsp/portable/write_wav_file.o audio/dsp/portable/write_wav_file_generic.o audio/tactile/util.o audio/tactile/energy_envelope/energy_envelope.o

RUN_AUTO_GAIN_CONTROL_ON_WAV_OBJS=audio/dsp/portable/run_auto_gain_control_on_wav.o audio/dsp/portable/auto_gain_control.o audio/dsp/portable/fast_fun.o audio/dsp/portable/read_wav_file.o audio/dsp/portable/read_wav_file_generic.o audio/dsp/portable/write_wav_file.o audio/dsp/portable/write_wav_file_generic.o audio/tactile/util.o

RUN_YUAN2005_OBJS=audio/tactile/references/yuan2005/run_yuan2005.o audio/tactile/references/yuan2005/yuan2005.o audio/dsp/portable/auto_gain_control.o audio/dsp/portable/biquad_filter.o audio/dsp/portable/butterworth.o audio/dsp/portable/complex.o audio/dsp/portable/fast_fun.o audio/dsp/portable/phasor_rotator.o audio/tactile/util.o

RUN_BRATAKOS2001_OBJS=audio/tactile/references/bratakos2001/run_bratakos2001.o audio/tactile/references/bratakos2001/bratakos2001.o audio/dsp/portable/auto_gain_control.o audio/dsp/portable/biquad_filter.o audio/dsp/portable/butterworth.o audio/dsp/portable/complex.o audio/dsp/portable/fast_fun.o audio/dsp/portable/phasor_rotator.o audio/tactile/util.o

# Benchmark.
TACTILE_PROCESSOR_BENCHMARK_OBJS=audio/tactile/tactile_processor_benchmark.o audio/tactile/tactile_processor.a

# Tests for audio/dsp/portable.
AUTO_GAIN_CONTROL_TEST_OBJS=audio/dsp/portable/auto_gain_control_test.o audio/dsp/portable/auto_gain_control.o audio/dsp/portable/fast_fun.o

BUTTERWORTH_TEST_OBJS=audio/dsp/portable/butterworth_test.o audio/dsp/portable/complex.o audio/dsp/portable/butterworth.o

COMPLEX_TEST_OBJS=audio/dsp/portable/complex_test.o audio/dsp/portable/complex.o

ELLIPTIC_FUN_TEST_OBJS=audio/dsp/portable/elliptic_fun_test.o audio/dsp/portable/elliptic_fun.o audio/dsp/portable/complex.o

FAST_FUN_TEST_OBJS=audio/dsp/portable/fast_fun_test.o audio/dsp/portable/fast_fun.o audio/dsp/portable/fast_fun_compute_tables.o

IIR_DESIGN_TEST_OBJS=audio/dsp/portable/iir_design_test.o audio/dsp/portable/iir_design.o audio/dsp/portable/complex.o audio/dsp/portable/elliptic_fun.o

MATH_CONSTANTS_TEST_OBJS=audio/dsp/portable/math_constants_test.o

PHASOR_ROTATOR_TEST_OBJS=audio/dsp/portable/phasor_rotator_test.o audio/dsp/portable/phasor_rotator.o

READ_WAV_FILE_TEST_OBJS=audio/dsp/portable/read_wav_file_test.o audio/dsp/portable/read_wav_file.o audio/dsp/portable/read_wav_file_generic.o audio/dsp/portable/write_wav_file.o audio/dsp/portable/write_wav_file_generic.o

READ_WAV_FILE_GENERIC_TEST_OBJS=audio/dsp/portable/read_wav_file_generic_test.o audio/dsp/portable/read_wav_file_generic.o

SERIALIZE_TEST_OBJS=audio/dsp/portable/serialize_test.o

WRITE_WAV_FILE_TEST_OBJS=audio/dsp/portable/write_wav_file_test.o audio/dsp/portable/write_wav_file.o audio/dsp/portable/write_wav_file_generic.o

# Tests for audio/tactile.
CARL_FRONTEND_TEST_OBJS=audio/tactile/frontend/carl_frontend_test.o audio/tactile/frontend/carl_frontend.o audio/tactile/frontend/carl_frontend_design.o audio/dsp/portable/complex.o audio/dsp/portable/fast_fun.o

EMBED_VOWEL_TEST_OBJS=audio/tactile/phone_embedding/embed_vowel_test.o audio/tactile/phone_embedding/embed_vowel.o audio/dsp/portable/butterworth.o audio/dsp/portable/complex.o audio/dsp/portable/fast_fun.o audio/dsp/portable/read_wav_file.o audio/dsp/portable/read_wav_file_generic.o audio/tactile/frontend/carl_frontend.o audio/tactile/frontend/carl_frontend_design.o audio/tactile/hexagon_interpolation.o audio/tactile/nn_ops.o

PHONEME_CODE_TEST_OBJS=audio/tactile/references/taps/phoneme_code.o audio/tactile/util.o audio/tactile/references/taps/phoneme_code_test.o

TACTOPHONE_LESSON_TEST_OBJS=audio/tactile/references/taps/tactophone_lesson_test.o audio/tactile/references/taps/tactophone_lesson.o

TACTOPHONE_ENGINE_TEST_OBJS=audio/tactile/references/taps/tactophone_engine_test.o audio/tactile/references/taps/phoneme_code.o audio/tactile/references/taps/tactophone_lesson.o audio/tactile/references/taps/tactophone_engine.o audio/tactile/references/taps/tactile_player.o audio/tactile/util.o audio/tactile/channel_map.o

ENERGY_ENVELOPE_TEST_OBJS=audio/tactile/energy_envelope/energy_envelope_test.o audio/tactile/energy_envelope/energy_envelope.o audio/dsp/portable/butterworth.o audio/dsp/portable/complex.o audio/dsp/portable/fast_fun.o

YUAN2005_TEST_OBJS=audio/tactile/references/yuan2005/yuan2005_test.o audio/tactile/references/yuan2005/yuan2005.o audio/dsp/portable/biquad_filter.o audio/dsp/portable/butterworth.o audio/dsp/portable/complex.o audio/dsp/portable/phasor_rotator.o

CHANNEL_MAP_TEST_OBJS=audio/tactile/channel_map_test.o audio/tactile/channel_map.o audio/tactile/util.o

HEXAGON_INTERPOLATION_TEST_OBJS=audio/tactile/hexagon_interpolation_test.o audio/tactile/hexagon_interpolation.o

NN_OPS_TEST_OBJS=audio/tactile/nn_ops_test.o audio/tactile/nn_ops.o

TACTILE_PLAYER_TEST_OBJS=audio/tactile/references/taps/tactile_player_test.o audio/tactile/references/taps/tactile_player.o

TACTILE_PROCESSOR_TEST_OBJS=audio/tactile/tactile_processor_test.o audio/dsp/portable/read_wav_file.o audio/dsp/portable/read_wav_file_generic.o audio/tactile/tactile_processor.a

UTIL_TEST_OBJS=audio/tactile/util_test.o audio/tactile/util.o

MD_FILES = $(shell find . -type f -name '*.md')
HTML_FILES = $(patsubst %.md, %.html, $(MD_FILES))

.PHONY: all check clean default dist doc
.SUFFIXES: .c .o .a .PICo .PICa .md .html
default: $(PROGRAMS)
all: $(PROGRAMS) $(BENCHMARKS) $(TESTS)

run_tactile_processor: $(RUN_TACTILE_PROCESSOR_OBJS)
	$(CC) $(RUN_TACTILE_PROCESSOR_OBJS) -lSDL2 -lportaudio $(LDFLAGS) -o $@

energy_envelope.so: $(ENERGY_ENVELOPE_PYTHON_BINDINGS_OBJS)
	$(CC) $(ENERGY_ENVELOPE_PYTHON_BINDINGS_OBJS) -shared -fPIC $(LDFLAGS) -o $@

tactile_processor.so: $(TACTILE_PROCESSOR_PYTHON_BINDINGS_OBJS)
	$(CC) $(TACTILE_PROCESSOR_PYTHON_BINDINGS_OBJS) -shared -fPIC $(LDFLAGS) -o $@

tactile_worker.so: $(TACTILE_WORKER_PYTHON_BINDINGS_OBJS)
	$(CC) $(TACTILE_WORKER_PYTHON_BINDINGS_OBJS) -shared -fPIC -lportaudio $(LDFLAGS) -o $@

tactophone: $(TACTOPHONE_OBJS)
	$(CC) $(TACTOPHONE_OBJS) -lportaudio -lncurses -pthread $(LDFLAGS) -o $@

tactometer: $(TACTOMETER_OBJS)
	$(CC) $(TACTOMETER_OBJS) -lSDL2 -lportaudio $(LDFLAGS) -o $@

play_buzz: $(PLAY_BUZZ_OBJS)
	$(CC) $(PLAY_BUZZ_OBJS) -lportaudio -lncurses $(LDFLAGS) -o $@

run_energy_envelope: $(RUN_ENERGY_ENVELOPE_OBJS)
	$(CC) $(RUN_ENERGY_ENVELOPE_OBJS) -lportaudio $(LDFLAGS) -o $@

run_energy_envelope_on_wav: $(RUN_ENERGY_ENVELOPE_ON_WAV_OBJS)
	$(CC) $(RUN_ENERGY_ENVELOPE_ON_WAV_OBJS) $(LDFLAGS) -o $@

run_auto_gain_control_on_wav: $(RUN_AUTO_GAIN_CONTROL_ON_WAV_OBJS)
	$(CC) $(RUN_AUTO_GAIN_CONTROL_ON_WAV_OBJS) $(LDFLAGS) -o $@

run_yuan2005: $(RUN_YUAN2005_OBJS)
	$(CC) $(RUN_YUAN2005_OBJS) -lportaudio $(LDFLAGS) -o $@

run_bratakos2001: $(RUN_BRATAKOS2001_OBJS)
	$(CC) $(RUN_BRATAKOS2001_OBJS) -lportaudio $(LDFLAGS) -o $@

tactile_processor_benchmark: $(TACTILE_PROCESSOR_BENCHMARK_OBJS)
	$(CC) $(TACTILE_PROCESSOR_BENCHMARK_OBJS) $(LDFLAGS) -o $@

auto_gain_control_test: $(AUTO_GAIN_CONTROL_TEST_OBJS)
	$(CC) $(AUTO_GAIN_CONTROL_TEST_OBJS) $(LDFLAGS) -o $@

butterworth_test: $(BUTTERWORTH_TEST_OBJS)
	$(CC) $(BUTTERWORTH_TEST_OBJS) $(LDFLAGS) -o $@

complex_test: $(COMPLEX_TEST_OBJS)
	$(CC) $(COMPLEX_TEST_OBJS) $(LDFLAGS) -o $@

elliptic_fun_test: $(ELLIPTIC_FUN_TEST_OBJS)
	$(CC) $(ELLIPTIC_FUN_TEST_OBJS) $(LDFLAGS) -o $@

fast_fun_test: $(FAST_FUN_TEST_OBJS)
	$(CC) $(FAST_FUN_TEST_OBJS) $(LDFLAGS) -o $@

iir_design_test: $(IIR_DESIGN_TEST_OBJS)
	$(CC) $(IIR_DESIGN_TEST_OBJS) $(LDFLAGS) -o $@

math_constants_test: $(MATH_CONSTANTS_TEST_OBJS)
	$(CC) $(MATH_CONSTANTS_TEST_OBJS) $(LDFLAGS) -o $@

phasor_rotator_test: $(PHASOR_ROTATOR_TEST_OBJS)
	$(CC) $(PHASOR_ROTATOR_TEST_OBJS) $(LDFLAGS) -o $@

read_wav_file_test: $(READ_WAV_FILE_TEST_OBJS)
	$(CC) $(READ_WAV_FILE_TEST_OBJS) $(LDFLAGS) -o $@

read_wav_file_generic_test: $(READ_WAV_FILE_GENERIC_TEST_OBJS)
	$(CC) $(READ_WAV_FILE_GENERIC_TEST_OBJS) $(LDFLAGS) -o $@

serialize_test: $(SERIALIZE_TEST_OBJS)
	$(CC) $(SERIALIZE_TEST_OBJS) $(LDFLAGS) -o $@

write_wav_file_test: $(WRITE_WAV_FILE_TEST_OBJS)
	$(CC) $(WRITE_WAV_FILE_TEST_OBJS) $(LDFLAGS) -o $@

carl_frontend_test: $(CARL_FRONTEND_TEST_OBJS)
	$(CC) $(CARL_FRONTEND_TEST_OBJS) $(LDFLAGS) -o $@

embed_vowel_test: $(EMBED_VOWEL_TEST_OBJS)
	$(CC) $(EMBED_VOWEL_TEST_OBJS) $(LDFLAGS) -o $@

phoneme_code_test: $(PHONEME_CODE_TEST_OBJS)
	$(CC) $(PHONEME_CODE_TEST_OBJS) $(LDFLAGS) -o $@

tactophone_engine_test: $(TACTOPHONE_ENGINE_TEST_OBJS)
	$(CC) $(TACTOPHONE_ENGINE_TEST_OBJS) -lportaudio -lncurses -pthread $(LDFLAGS) -o $@

tactophone_lesson_test: $(TACTOPHONE_LESSON_TEST_OBJS)
	$(CC) $(TACTOPHONE_LESSON_TEST_OBJS) $(LDFLAGS) -o $@

energy_envelope_test: $(ENERGY_ENVELOPE_TEST_OBJS)
	$(CC) $(ENERGY_ENVELOPE_TEST_OBJS) $(LDFLAGS) -o $@

channel_map_test: $(CHANNEL_MAP_TEST_OBJS)
	$(CC) $(CHANNEL_MAP_TEST_OBJS) $(LDFLAGS) -o $@

hexagon_interpolation_test: $(HEXAGON_INTERPOLATION_TEST_OBJS)
	$(CC) $(HEXAGON_INTERPOLATION_TEST_OBJS) $(LDFLAGS) -o $@

nn_ops_test: $(NN_OPS_TEST_OBJS)
	$(CC) $(NN_OPS_TEST_OBJS) $(LDFLAGS) -o $@

tactile_player_test: $(TACTILE_PLAYER_TEST_OBJS)
	$(CC) $(TACTILE_PLAYER_TEST_OBJS) -pthread $(LDFLAGS) -o $@

tactile_processor_test: $(TACTILE_PROCESSOR_TEST_OBJS)
	$(CC) $(TACTILE_PROCESSOR_TEST_OBJS) $(LDFLAGS) -o $@

util_test: $(UTIL_TEST_OBJS)
	$(CC) $(UTIL_TEST_OBJS) $(LDFLAGS) -o $@

yuan2005_test: $(YUAN2005_TEST_OBJS)
	$(CC) $(YUAN2005_TEST_OBJS) $(LDFLAGS) -o $@

audio/tactile/tactile_processor.a: audio/tactile/tactile_processor.a(audio/tactile/tactile_processor.o) audio/tactile/tactile_processor.a(audio/tactile/phone_embedding/embed_vowel.o) audio/tactile/tactile_processor.a(audio/tactile/energy_envelope/energy_envelope.o) audio/tactile/tactile_processor.a(audio/tactile/hexagon_interpolation.o) audio/tactile/tactile_processor.a(audio/tactile/post_processor.o) audio/tactile/tactile_processor.a(audio/tactile/tactor_equalizer.o) audio/tactile/tactile_processor.a(audio/tactile/nn_ops.o) audio/tactile/tactile_processor.a(audio/tactile/frontend/carl_frontend.o) audio/tactile/tactile_processor.a(audio/tactile/frontend/carl_frontend_design.o) audio/tactile/tactile_processor.a(audio/dsp/portable/biquad_filter.o) audio/tactile/tactile_processor.a(audio/dsp/portable/butterworth.o) audio/tactile/tactile_processor.a(audio/dsp/portable/complex.o) audio/tactile/tactile_processor.a(audio/dsp/portable/fast_fun.o)

audio/tactile/tactile_processor.PICa: audio/tactile/tactile_processor.PICa(audio/tactile/tactile_processor.PICo) audio/tactile/tactile_processor.PICa(audio/tactile/phone_embedding/embed_vowel.PICo) audio/tactile/tactile_processor.PICa(audio/tactile/energy_envelope/energy_envelope.PICo) audio/tactile/tactile_processor.PICa(audio/dsp/portable/biquad_filter.PICo) audio/tactile/tactile_processor.PICa(audio/tactile/hexagon_interpolation.PICo) audio/tactile/tactile_processor.PICa(audio/tactile/post_processor.PICo) audio/tactile/tactile_processor.PICa(audio/tactile/tactor_equalizer.PICo) audio/tactile/tactile_processor.PICa(audio/tactile/nn_ops.PICo) audio/tactile/tactile_processor.PICa(audio/tactile/frontend/carl_frontend.PICo) audio/tactile/tactile_processor.PICa(audio/tactile/frontend/carl_frontend_design.PICo) audio/tactile/tactile_processor.PICa(audio/dsp/portable/biquad_filter.PICo) audio/tactile/tactile_processor.PICa(audio/dsp/portable/butterworth.PICo) audio/tactile/tactile_processor.PICa(audio/dsp/portable/complex.PICo) audio/tactile/tactile_processor.PICa(audio/dsp/portable/fast_fun.PICo)

audio/tactile/python/energy_envelope_python_bindings.PICo: audio/tactile/python/energy_envelope_python_bindings.c
	$(CC) -fPIC $(PYTHON_BINDINGS_CFLAGS) -c -o $@ $<

audio/tactile/python/tactile_processor_python_bindings.PICo: audio/tactile/python/tactile_processor_python_bindings.c
	$(CC) -fPIC $(PYTHON_BINDINGS_CFLAGS) -c -o $@ $<

audio/tactile/python/tactile_worker_python_bindings.PICo: audio/tactile/python/tactile_worker_python_bindings.c
	$(CC) -fPIC $(PYTHON_BINDINGS_CFLAGS) -c -o $@ $<

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<

.c.PICo:
	$(CC) -fPIC $(CFLAGS) -c -o $@ $<

check: $(TESTS)
	for name in $(TESTS); do echo ./$$name; ./$$name || echo -e "\n\033[01;31mFAILED\033[00m: $$name\n"; done

clean:
	$(RM) -f -- $(RUN_TACTILE_PROCESSOR_OBJS) $(ENERGY_ENVELOPE_PYTHON_BINDINGS_OBJS) $(TACTILE_PROCESSOR_PYTHON_BINDINGS_OBJS) $(TACTILE_WORKER_PYTHON_BINDINGS_OBJS) $(TACTOPHONE_OBJS) $(TACTOMETER_OBJS) $(PLAY_BUZZ_OBJS) $(AUTO_GAIN_CONTROL_TEST_OBJS) $(RUN_YUAN2005_OBJS) $(RUN_BRATAKOS2001_OBJS) $(TACTILE_PROCESSOR_BENCHMARK_OBJS) $(RUN_ENERGY_ENVELOPE_OBJS) $(AUTO_GAIN_CONTROL_TEST_OBJS) $(BUTTERWORTH_TEST_OBJS) $(COMPLEX_TEST_OBJS) $(ELLIPTIC_FUN_TEST_OBJS) $(FAST_FUN_TEST_OBJS) $(IIR_DESIGN_TEST_OBJS) $(MATH_CONSTANTS_TEST_OBJS) $(PHASOR_ROTATOR_TEST_OBJS) $(READ_WAV_FILE_TEST_OBJS) $(READ_WAV_FILE_GENERIC_TEST_OBJS) $(SERIALIZE_TEST_OBJS) $(WRITE_WAV_FILE_TEST_OBJS) $(CARL_FRONTEND_TEST_OBJS) $(EMBED_VOWEL_TEST_OBJS) $(TACTILE_PLAYER_TEST_OBJS) $(UTIL_TEST_OBJS) $(PHONEME_CODE_TEST_OBJS) $(TACTOPHONE_LESSON_TEST_OBJS) $(TACTOPHONE_ENGINE_TEST_OBJS) $(ENERGY_ENVELOPE_TEST_OBJS) $(CHANNEL_MAP_TEST_OBJS) $(HEXAGON_INTERPOLATION_TEST_OBJS) $(NN_OPS_TEST_OBJS) $(YUAN2005_TEST_OBJS) $(TACTILE_PROCESSOR_TEST_OBJS) $(PROGRAMS) $(BENCHMARKS) $(TESTS) audio/tactile/tactile_processor.a

doc: $(HTML_FILES)

%.html: %.md
		pandoc -s $< -o $@

