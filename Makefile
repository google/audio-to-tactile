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

CFLAGS += -O3 -g -I. -I./src -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-sign-compare
LDFLAGS += -lm

# Compile flags for Python bindings. Set Python version for your installation.
PYTHON_BINDINGS_CFLAGS = \
		$$(pkg-config --cflags python-3.8) -Wno-missing-field-initializers -Wno-cast-function-type $(CFLAGS)

PROGRAMS=run_tactile_processor energy_envelope.so tactile_processor.so tactile_worker.so tactophone tactometer play_buzz run_energy_envelope run_energy_envelope_on_wav run_auto_gain_control_on_wav run_yuan2005 run_bratakos2001
BENCHMARKS=tactile_processor_benchmark
TESTS=auto_gain_control_test butterworth_test complex_test elliptic_fun_test fast_fun_test math_constants_test phasor_rotator_test read_wav_file_test read_wav_file_generic_test serialize_test write_wav_file_test carl_frontend_test embed_vowel_test phoneme_code_test tactophone_engine_test tactophone_lesson_test energy_envelope_test channel_map_test hexagon_interpolation_test nn_ops_test tactile_player_test tactile_processor_test util_test yuan2005_test

RUN_TACTILE_PROCESSOR_OBJS=extras/tools/run_tactile_processor.o extras/tools/run_tactile_processor_assets.o src/dsp/number_util.o src/dsp/read_wav_file.o src/dsp/read_wav_file_generic.o extras/tools/channel_map.o extras/tools/portaudio_device.o extras/tools/util.o extras/tools/sdl/basic_sdl_app.o extras/tools/sdl/texture_from_rle_data.o extras/tools/sdl/window_icon.o tactile_processor.a

ENERGY_ENVELOPE_PYTHON_BINDINGS_OBJS=extras/python/tactile/energy_envelope_python_bindings.PICo src/tactile/energy_envelope.PICo src/dsp/butterworth.PICo src/dsp/complex.PICo src/dsp/fast_fun.PICo

TACTILE_PROCESSOR_PYTHON_BINDINGS_OBJS=extras/python/tactile/tactile_processor_python_bindings.PICo tactile_processor.PICa

TACTILE_WORKER_PYTHON_BINDINGS_OBJS=extras/python/tactile/tactile_worker_python_bindings.PICo extras/python/tactile/tactile_worker.PICo tactile_processor.PICa extras/tools/channel_map.PICo extras/tools/portaudio_device.PICo extras/tools/util.PICo

TACTOPHONE_OBJS=extras/references/taps/tactophone_main.o extras/references/taps/tactophone_state_main_menu.o extras/references/taps/tactophone_state_free_play.o extras/references/taps/tactophone_state_test_tactors.o extras/references/taps/tactophone_state_begin_lesson.o extras/references/taps/tactophone_state_lesson_trial.o extras/references/taps/tactophone_state_lesson_review.o extras/references/taps/tactophone_state_lesson_done.o extras/references/taps/phoneme_code.o extras/references/taps/tactophone_engine.o extras/references/taps/tactophone.o extras/references/taps/tactophone_lesson.o extras/tools/util.o extras/references/taps/tactile_player.o extras/tools/channel_map.o

TACTOMETER_OBJS=extras/tools/tactometer.o extras/tools/portaudio_device.o extras/tools/sdl/basic_sdl_app.o extras/tools/sdl/draw_text.o extras/tools/sdl/window_icon.o extras/tools/util.o

PLAY_BUZZ_OBJS=extras/tools/play_buzz.o extras/tools/portaudio_device.o extras/tools/util.o

RUN_ENERGY_ENVELOPE_OBJS=extras/tools/run_energy_envelope.o src/dsp/butterworth.o src/dsp/complex.o src/dsp/fast_fun.o extras/tools/portaudio_device.o extras/tools/util.o src/tactile/energy_envelope.o

RUN_ENERGY_ENVELOPE_ON_WAV_OBJS=extras/tools/run_energy_envelope_on_wav.o src/dsp/butterworth.o src/dsp/complex.o src/dsp/fast_fun.o src/dsp/read_wav_file.o src/dsp/read_wav_file_generic.o src/dsp/write_wav_file.o src/dsp/write_wav_file_generic.o extras/tools/util.o src/tactile/energy_envelope.o

RUN_AUTO_GAIN_CONTROL_ON_WAV_OBJS=extras/tools/run_auto_gain_control_on_wav.o src/dsp/auto_gain_control.o src/dsp/fast_fun.o src/dsp/read_wav_file.o src/dsp/read_wav_file_generic.o src/dsp/write_wav_file.o src/dsp/write_wav_file_generic.o extras/tools/util.o

RUN_YUAN2005_OBJS=extras/references/yuan2005/run_yuan2005.o extras/references/yuan2005/yuan2005.o src/dsp/auto_gain_control.o src/dsp/biquad_filter.o src/dsp/butterworth.o src/dsp/complex.o src/dsp/fast_fun.o src/dsp/phasor_rotator.o extras/tools/util.o

RUN_BRATAKOS2001_OBJS=extras/references/bratakos2001/run_bratakos2001.o extras/references/bratakos2001/bratakos2001.o src/dsp/auto_gain_control.o src/dsp/biquad_filter.o src/dsp/butterworth.o src/dsp/complex.o src/dsp/fast_fun.o src/dsp/phasor_rotator.o extras/tools/util.o

# Benchmark.
TACTILE_PROCESSOR_BENCHMARK_OBJS=extras/tools/tactile_processor_benchmark.o tactile_processor.a

# Tests for dsp.
AUTO_GAIN_CONTROL_TEST_OBJS=extras/test/dsp/auto_gain_control_test.o src/dsp/auto_gain_control.o src/dsp/fast_fun.o

BUTTERWORTH_TEST_OBJS=extras/test/dsp/butterworth_test.o src/dsp/complex.o src/dsp/butterworth.o

COMPLEX_TEST_OBJS=extras/test/dsp/complex_test.o src/dsp/complex.o

ELLIPTIC_FUN_TEST_OBJS=extras/test/dsp/elliptic_fun_test.o src/dsp/elliptic_fun.o src/dsp/complex.o

FAST_FUN_TEST_OBJS=extras/test/dsp/fast_fun_test.o src/dsp/fast_fun.o src/dsp/fast_fun_compute_tables.o

IIR_DESIGN_TEST_OBJS=extras/test/dsp/iir_design_test.o src/dsp/iir_design.o src/dsp/complex.o src/dsp/elliptic_fun.o

MATH_CONSTANTS_TEST_OBJS=extras/test/dsp/math_constants_test.o

PHASOR_ROTATOR_TEST_OBJS=extras/test/dsp/phasor_rotator_test.o src/dsp/phasor_rotator.o

READ_WAV_FILE_TEST_OBJS=extras/test/dsp/read_wav_file_test.o src/dsp/read_wav_file.o src/dsp/read_wav_file_generic.o src/dsp/write_wav_file.o src/dsp/write_wav_file_generic.o

READ_WAV_FILE_GENERIC_TEST_OBJS=extras/test/dsp/read_wav_file_generic_test.o src/dsp/read_wav_file_generic.o

SERIALIZE_TEST_OBJS=extras/test/dsp/serialize_test.o

WRITE_WAV_FILE_TEST_OBJS=extras/test/dsp/write_wav_file_test.o src/dsp/write_wav_file.o src/dsp/write_wav_file_generic.o

# Tests for tactile.
CARL_FRONTEND_TEST_OBJS=extras/test/frontend/carl_frontend_test.o src/frontend/carl_frontend.o src/frontend/carl_frontend_design.o src/dsp/complex.o src/dsp/fast_fun.o

EMBED_VOWEL_TEST_OBJS=extras/test/phonetics/embed_vowel_test.o src/phonetics/embed_vowel.o src/dsp/butterworth.o src/dsp/complex.o src/dsp/fast_fun.o src/dsp/read_wav_file.o src/dsp/read_wav_file_generic.o src/frontend/carl_frontend.o src/frontend/carl_frontend_design.o src/phonetics/hexagon_interpolation.o src/phonetics/nn_ops.o

PHONEME_CODE_TEST_OBJS=extras/references/taps/phoneme_code.o extras/tools/util.o extras/references/taps/phoneme_code_test.o

TACTOPHONE_LESSON_TEST_OBJS=extras/references/taps/tactophone_lesson_test.o extras/references/taps/tactophone_lesson.o

TACTOPHONE_ENGINE_TEST_OBJS=extras/references/taps/tactophone_engine_test.o extras/references/taps/phoneme_code.o extras/references/taps/tactophone_lesson.o extras/references/taps/tactophone_engine.o extras/references/taps/tactile_player.o extras/tools/util.o extras/tools/channel_map.o

ENERGY_ENVELOPE_TEST_OBJS=extras/test/tactile/energy_envelope_test.o src/tactile/energy_envelope.o src/dsp/butterworth.o src/dsp/complex.o src/dsp/fast_fun.o

YUAN2005_TEST_OBJS=extras/references/yuan2005/yuan2005_test.o extras/references/yuan2005/yuan2005.o src/dsp/biquad_filter.o src/dsp/butterworth.o src/dsp/complex.o src/dsp/phasor_rotator.o

CHANNEL_MAP_TEST_OBJS=extras/tools/channel_map_test.o extras/tools/channel_map.o extras/tools/util.o

HEXAGON_INTERPOLATION_TEST_OBJS=extras/test/phonetics/hexagon_interpolation_test.o src/phonetics/hexagon_interpolation.o

NN_OPS_TEST_OBJS=extras/test/phonetics/nn_ops_test.o src/phonetics/nn_ops.o src/dsp/fast_fun.o

TACTILE_PLAYER_TEST_OBJS=extras/references/taps/tactile_player_test.o extras/references/taps/tactile_player.o

TACTILE_PROCESSOR_TEST_OBJS=extras/test/tactile/tactile_processor_test.o src/dsp/read_wav_file.o src/dsp/read_wav_file_generic.o tactile_processor.a

UTIL_TEST_OBJS=extras/tools/util_test.o extras/tools/util.o

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

tactile_processor.a: tactile_processor.a(src/tactile/tactile_processor.o) tactile_processor.a(src/phonetics/embed_vowel.o) tactile_processor.a(src/tactile/energy_envelope.o) tactile_processor.a(src/phonetics/hexagon_interpolation.o) tactile_processor.a(src/tactile/post_processor.o) tactile_processor.a(src/tactile/tactor_equalizer.o) tactile_processor.a(src/phonetics/nn_ops.o) tactile_processor.a(src/frontend/carl_frontend.o) tactile_processor.a(src/frontend/carl_frontend_design.o) tactile_processor.a(src/dsp/biquad_filter.o) tactile_processor.a(src/dsp/butterworth.o) tactile_processor.a(src/dsp/complex.o) tactile_processor.a(src/dsp/fast_fun.o)

tactile_processor.PICa: tactile_processor.PICa(src/tactile/tactile_processor.PICo) tactile_processor.PICa(src/phonetics/embed_vowel.PICo) tactile_processor.PICa(src/tactile/energy_envelope.PICo) tactile_processor.PICa(src/phonetics/hexagon_interpolation.PICo) tactile_processor.PICa(src/tactile/post_processor.PICo) tactile_processor.PICa(src/tactile/tactor_equalizer.PICo) tactile_processor.PICa(src/phonetics/nn_ops.PICo) tactile_processor.PICa(src/frontend/carl_frontend.PICo) tactile_processor.PICa(src/frontend/carl_frontend_design.PICo) tactile_processor.PICa(src/dsp/biquad_filter.PICo) tactile_processor.PICa(src/dsp/butterworth.PICo) tactile_processor.PICa(src/dsp/complex.PICo) tactile_processor.PICa(src/dsp/fast_fun.PICo)

extras/python/tactile/energy_envelope_python_bindings.PICo: extras/python/tactile/energy_envelope_python_bindings.c
	$(CC) -fPIC $(PYTHON_BINDINGS_CFLAGS) -c -o $@ $<

extras/python/tactile/tactile_processor_python_bindings.PICo: extras/python/tactile/tactile_processor_python_bindings.c
	$(CC) -fPIC $(PYTHON_BINDINGS_CFLAGS) -c -o $@ $<

extras/python/tactile/tactile_worker_python_bindings.PICo: extras/python/tactile/tactile_worker_python_bindings.c
	$(CC) -fPIC $(PYTHON_BINDINGS_CFLAGS) -c -o $@ $<

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<

.c.PICo:
	$(CC) -fPIC $(CFLAGS) -c -o $@ $<

check: $(TESTS)
	for name in $(TESTS); do echo ./$$name; ./$$name || echo -e "\n\033[01;31mFAILED\033[00m: $$name\n"; done
	@echo -e "\n**** All tests pass ****\n"

clean:
	$(RM) -f -- $(RUN_TACTILE_PROCESSOR_OBJS) $(ENERGY_ENVELOPE_PYTHON_BINDINGS_OBJS) $(TACTILE_PROCESSOR_PYTHON_BINDINGS_OBJS) $(TACTILE_WORKER_PYTHON_BINDINGS_OBJS) $(TACTOPHONE_OBJS) $(TACTOMETER_OBJS) $(PLAY_BUZZ_OBJS) $(AUTO_GAIN_CONTROL_TEST_OBJS) $(RUN_YUAN2005_OBJS) $(RUN_BRATAKOS2001_OBJS) $(TACTILE_PROCESSOR_BENCHMARK_OBJS) $(RUN_ENERGY_ENVELOPE_OBJS) $(AUTO_GAIN_CONTROL_TEST_OBJS) $(BUTTERWORTH_TEST_OBJS) $(COMPLEX_TEST_OBJS) $(ELLIPTIC_FUN_TEST_OBJS) $(FAST_FUN_TEST_OBJS) $(IIR_DESIGN_TEST_OBJS) $(MATH_CONSTANTS_TEST_OBJS) $(PHASOR_ROTATOR_TEST_OBJS) $(READ_WAV_FILE_TEST_OBJS) $(READ_WAV_FILE_GENERIC_TEST_OBJS) $(SERIALIZE_TEST_OBJS) $(WRITE_WAV_FILE_TEST_OBJS) $(CARL_FRONTEND_TEST_OBJS) $(EMBED_VOWEL_TEST_OBJS) $(TACTILE_PLAYER_TEST_OBJS) $(UTIL_TEST_OBJS) $(PHONEME_CODE_TEST_OBJS) $(TACTOPHONE_LESSON_TEST_OBJS) $(TACTOPHONE_ENGINE_TEST_OBJS) $(ENERGY_ENVELOPE_TEST_OBJS) $(CHANNEL_MAP_TEST_OBJS) $(HEXAGON_INTERPOLATION_TEST_OBJS) $(NN_OPS_TEST_OBJS) $(YUAN2005_TEST_OBJS) $(TACTILE_PROCESSOR_TEST_OBJS) $(PROGRAMS) $(BENCHMARKS) $(TESTS) tactile_processor.a tactile_processor.PICa

doc: $(HTML_FILES)

%.html: %.md
		pandoc -s $< -o $@

