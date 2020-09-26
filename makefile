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
		$$(pkg-config --cflags python-3.8) -Wno-missing-field-initializers -Wno-cast-function-type $(CFLAGS)

PROGRAMS=run_tactile_processor energy_envelope.so tactile_processor.so tactile_worker.so tactophone tactometer play_buzz run_energy_envelope run_energy_envelope_on_wav run_auto_gain_control_on_wav run_yuan2005 run_bratakos2001
BENCHMARKS=tactile_processor_benchmark
TESTS=auto_gain_control_test butterworth_test complex_test elliptic_fun_test fast_fun_test math_constants_test phasor_rotator_test read_wav_file_test read_wav_file_generic_test serialize_test write_wav_file_test carl_frontend_test embed_vowel_test phoneme_code_test tactophone_engine_test tactophone_lesson_test energy_envelope_test channel_map_test hexagon_interpolation_test nn_ops_test tactile_player_test tactile_processor_test util_test yuan2005_test

RUN_TACTILE_PROCESSOR_OBJS=tactile/run_tactile_processor.o tactile/run_tactile_processor_assets.o dsp/number_util.o dsp/read_wav_file.o dsp/read_wav_file_generic.o tactile/channel_map.o tactile/portaudio_device.o tactile/util.o sdl/basic_sdl_app.o sdl/texture_from_rle_data.o sdl/window_icon.o tactile/tactile_processor.a

ENERGY_ENVELOPE_PYTHON_BINDINGS_OBJS=tactile/python/energy_envelope_python_bindings.PICo tactile/energy_envelope/energy_envelope.PICo dsp/butterworth.PICo dsp/complex.PICo dsp/fast_fun.PICo

TACTILE_PROCESSOR_PYTHON_BINDINGS_OBJS=tactile/python/tactile_processor_python_bindings.PICo tactile/tactile_processor.PICa

TACTILE_WORKER_PYTHON_BINDINGS_OBJS=tactile/python/tactile_worker_python_bindings.PICo tactile/python/tactile_worker.PICo tactile/tactile_processor.PICa tactile/channel_map.PICo tactile/portaudio_device.PICo tactile/util.PICo

TACTOPHONE_OBJS=tactile/references/taps/tactophone_main.o tactile/references/taps/tactophone_state_main_menu.o tactile/references/taps/tactophone_state_free_play.o tactile/references/taps/tactophone_state_test_tactors.o tactile/references/taps/tactophone_state_begin_lesson.o tactile/references/taps/tactophone_state_lesson_trial.o tactile/references/taps/tactophone_state_lesson_review.o tactile/references/taps/tactophone_state_lesson_done.o tactile/references/taps/phoneme_code.o tactile/references/taps/tactophone_engine.o tactile/references/taps/tactophone.o tactile/references/taps/tactophone_lesson.o tactile/util.o tactile/references/taps/tactile_player.o tactile/channel_map.o

TACTOMETER_OBJS=tactile/tactometer.o tactile/portaudio_device.o sdl/basic_sdl_app.o sdl/draw_text.o sdl/window_icon.o tactile/util.o

PLAY_BUZZ_OBJS=tactile/play_buzz.o tactile/util.o

RUN_ENERGY_ENVELOPE_OBJS=tactile/energy_envelope/run_energy_envelope.o dsp/butterworth.o dsp/complex.o dsp/fast_fun.o tactile/portaudio_device.o tactile/util.o tactile/energy_envelope/energy_envelope.o

RUN_ENERGY_ENVELOPE_ON_WAV_OBJS=tactile/energy_envelope/run_energy_envelope_on_wav.o dsp/butterworth.o dsp/complex.o dsp/fast_fun.o dsp/read_wav_file.o dsp/read_wav_file_generic.o dsp/write_wav_file.o dsp/write_wav_file_generic.o tactile/util.o tactile/energy_envelope/energy_envelope.o

RUN_AUTO_GAIN_CONTROL_ON_WAV_OBJS=dsp/run_auto_gain_control_on_wav.o dsp/auto_gain_control.o dsp/fast_fun.o dsp/read_wav_file.o dsp/read_wav_file_generic.o dsp/write_wav_file.o dsp/write_wav_file_generic.o tactile/util.o

RUN_YUAN2005_OBJS=tactile/references/yuan2005/run_yuan2005.o tactile/references/yuan2005/yuan2005.o dsp/auto_gain_control.o dsp/biquad_filter.o dsp/butterworth.o dsp/complex.o dsp/fast_fun.o dsp/phasor_rotator.o tactile/util.o

RUN_BRATAKOS2001_OBJS=tactile/references/bratakos2001/run_bratakos2001.o tactile/references/bratakos2001/bratakos2001.o dsp/auto_gain_control.o dsp/biquad_filter.o dsp/butterworth.o dsp/complex.o dsp/fast_fun.o dsp/phasor_rotator.o tactile/util.o

# Benchmark.
TACTILE_PROCESSOR_BENCHMARK_OBJS=tactile/tactile_processor_benchmark.o tactile/tactile_processor.a

# Tests for dsp.
AUTO_GAIN_CONTROL_TEST_OBJS=dsp/auto_gain_control_test.o dsp/auto_gain_control.o dsp/fast_fun.o

BUTTERWORTH_TEST_OBJS=dsp/butterworth_test.o dsp/complex.o dsp/butterworth.o

COMPLEX_TEST_OBJS=dsp/complex_test.o dsp/complex.o

ELLIPTIC_FUN_TEST_OBJS=dsp/elliptic_fun_test.o dsp/elliptic_fun.o dsp/complex.o

FAST_FUN_TEST_OBJS=dsp/fast_fun_test.o dsp/fast_fun.o dsp/fast_fun_compute_tables.o

IIR_DESIGN_TEST_OBJS=dsp/iir_design_test.o dsp/iir_design.o dsp/complex.o dsp/elliptic_fun.o

MATH_CONSTANTS_TEST_OBJS=dsp/math_constants_test.o

PHASOR_ROTATOR_TEST_OBJS=dsp/phasor_rotator_test.o dsp/phasor_rotator.o

READ_WAV_FILE_TEST_OBJS=dsp/read_wav_file_test.o dsp/read_wav_file.o dsp/read_wav_file_generic.o dsp/write_wav_file.o dsp/write_wav_file_generic.o

READ_WAV_FILE_GENERIC_TEST_OBJS=dsp/read_wav_file_generic_test.o dsp/read_wav_file_generic.o

SERIALIZE_TEST_OBJS=dsp/serialize_test.o

WRITE_WAV_FILE_TEST_OBJS=dsp/write_wav_file_test.o dsp/write_wav_file.o dsp/write_wav_file_generic.o

# Tests for tactile.
CARL_FRONTEND_TEST_OBJS=frontend/carl_frontend_test.o frontend/carl_frontend.o frontend/carl_frontend_design.o dsp/complex.o dsp/fast_fun.o

EMBED_VOWEL_TEST_OBJS=phonetics/embed_vowel_test.o phonetics/embed_vowel.o dsp/butterworth.o dsp/complex.o dsp/fast_fun.o dsp/read_wav_file.o dsp/read_wav_file_generic.o frontend/carl_frontend.o frontend/carl_frontend_design.o tactile/hexagon_interpolation.o phonetics/nn_ops.o

PHONEME_CODE_TEST_OBJS=tactile/references/taps/phoneme_code.o tactile/util.o tactile/references/taps/phoneme_code_test.o

TACTOPHONE_LESSON_TEST_OBJS=tactile/references/taps/tactophone_lesson_test.o tactile/references/taps/tactophone_lesson.o

TACTOPHONE_ENGINE_TEST_OBJS=tactile/references/taps/tactophone_engine_test.o tactile/references/taps/phoneme_code.o tactile/references/taps/tactophone_lesson.o tactile/references/taps/tactophone_engine.o tactile/references/taps/tactile_player.o tactile/util.o tactile/channel_map.o

ENERGY_ENVELOPE_TEST_OBJS=tactile/energy_envelope/energy_envelope_test.o tactile/energy_envelope/energy_envelope.o dsp/butterworth.o dsp/complex.o dsp/fast_fun.o

YUAN2005_TEST_OBJS=tactile/references/yuan2005/yuan2005_test.o tactile/references/yuan2005/yuan2005.o dsp/biquad_filter.o dsp/butterworth.o dsp/complex.o dsp/phasor_rotator.o

CHANNEL_MAP_TEST_OBJS=tactile/channel_map_test.o tactile/channel_map.o tactile/util.o

HEXAGON_INTERPOLATION_TEST_OBJS=tactile/hexagon_interpolation_test.o tactile/hexagon_interpolation.o

NN_OPS_TEST_OBJS=phonetics/nn_ops_test.o phonetics/nn_ops.o dsp/fast_fun.o

TACTILE_PLAYER_TEST_OBJS=tactile/references/taps/tactile_player_test.o tactile/references/taps/tactile_player.o

TACTILE_PROCESSOR_TEST_OBJS=tactile/tactile_processor_test.o dsp/read_wav_file.o dsp/read_wav_file_generic.o tactile/tactile_processor.a

UTIL_TEST_OBJS=tactile/util_test.o tactile/util.o

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

tactile/tactile_processor.a: tactile/tactile_processor.a(tactile/tactile_processor.o) tactile/tactile_processor.a(phonetics/embed_vowel.o) tactile/tactile_processor.a(tactile/energy_envelope/energy_envelope.o) tactile/tactile_processor.a(tactile/hexagon_interpolation.o) tactile/tactile_processor.a(tactile/post_processor.o) tactile/tactile_processor.a(tactile/tactor_equalizer.o) tactile/tactile_processor.a(phonetics/nn_ops.o) tactile/tactile_processor.a(frontend/carl_frontend.o) tactile/tactile_processor.a(frontend/carl_frontend_design.o) tactile/tactile_processor.a(dsp/biquad_filter.o) tactile/tactile_processor.a(dsp/butterworth.o) tactile/tactile_processor.a(dsp/complex.o) tactile/tactile_processor.a(dsp/fast_fun.o)

tactile/tactile_processor.PICa: tactile/tactile_processor.PICa(tactile/tactile_processor.PICo) tactile/tactile_processor.PICa(phonetics/embed_vowel.PICo) tactile/tactile_processor.PICa(tactile/energy_envelope/energy_envelope.PICo) tactile/tactile_processor.PICa(dsp/biquad_filter.PICo) tactile/tactile_processor.PICa(tactile/hexagon_interpolation.PICo) tactile/tactile_processor.PICa(tactile/post_processor.PICo) tactile/tactile_processor.PICa(tactile/tactor_equalizer.PICo) tactile/tactile_processor.PICa(phonetics/nn_ops.PICo) tactile/tactile_processor.PICa(frontend/carl_frontend.PICo) tactile/tactile_processor.PICa(frontend/carl_frontend_design.PICo) tactile/tactile_processor.PICa(dsp/biquad_filter.PICo) tactile/tactile_processor.PICa(dsp/butterworth.PICo) tactile/tactile_processor.PICa(dsp/complex.PICo) tactile/tactile_processor.PICa(dsp/fast_fun.PICo)

tactile/python/energy_envelope_python_bindings.PICo: tactile/python/energy_envelope_python_bindings.c
	$(CC) -fPIC $(PYTHON_BINDINGS_CFLAGS) -c -o $@ $<

tactile/python/tactile_processor_python_bindings.PICo: tactile/python/tactile_processor_python_bindings.c
	$(CC) -fPIC $(PYTHON_BINDINGS_CFLAGS) -c -o $@ $<

tactile/python/tactile_worker_python_bindings.PICo: tactile/python/tactile_worker_python_bindings.c
	$(CC) -fPIC $(PYTHON_BINDINGS_CFLAGS) -c -o $@ $<

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<

.c.PICo:
	$(CC) -fPIC $(CFLAGS) -c -o $@ $<

check: $(TESTS)
	for name in $(TESTS); do echo ./$$name; ./$$name || echo -e "\n\033[01;31mFAILED\033[00m: $$name\n"; done

clean:
	$(RM) -f -- $(RUN_TACTILE_PROCESSOR_OBJS) $(ENERGY_ENVELOPE_PYTHON_BINDINGS_OBJS) $(TACTILE_PROCESSOR_PYTHON_BINDINGS_OBJS) $(TACTILE_WORKER_PYTHON_BINDINGS_OBJS) $(TACTOPHONE_OBJS) $(TACTOMETER_OBJS) $(PLAY_BUZZ_OBJS) $(AUTO_GAIN_CONTROL_TEST_OBJS) $(RUN_YUAN2005_OBJS) $(RUN_BRATAKOS2001_OBJS) $(TACTILE_PROCESSOR_BENCHMARK_OBJS) $(RUN_ENERGY_ENVELOPE_OBJS) $(AUTO_GAIN_CONTROL_TEST_OBJS) $(BUTTERWORTH_TEST_OBJS) $(COMPLEX_TEST_OBJS) $(ELLIPTIC_FUN_TEST_OBJS) $(FAST_FUN_TEST_OBJS) $(IIR_DESIGN_TEST_OBJS) $(MATH_CONSTANTS_TEST_OBJS) $(PHASOR_ROTATOR_TEST_OBJS) $(READ_WAV_FILE_TEST_OBJS) $(READ_WAV_FILE_GENERIC_TEST_OBJS) $(SERIALIZE_TEST_OBJS) $(WRITE_WAV_FILE_TEST_OBJS) $(CARL_FRONTEND_TEST_OBJS) $(EMBED_VOWEL_TEST_OBJS) $(TACTILE_PLAYER_TEST_OBJS) $(UTIL_TEST_OBJS) $(PHONEME_CODE_TEST_OBJS) $(TACTOPHONE_LESSON_TEST_OBJS) $(TACTOPHONE_ENGINE_TEST_OBJS) $(ENERGY_ENVELOPE_TEST_OBJS) $(CHANNEL_MAP_TEST_OBJS) $(HEXAGON_INTERPOLATION_TEST_OBJS) $(NN_OPS_TEST_OBJS) $(YUAN2005_TEST_OBJS) $(TACTILE_PROCESSOR_TEST_OBJS) $(PROGRAMS) $(BENCHMARKS) $(TESTS) tactile/tactile_processor.a

doc: $(HTML_FILES)

%.html: %.md
		pandoc -s $< -o $@

