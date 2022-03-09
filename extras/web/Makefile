# Copyright 2020-2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
#
# Makefile for web demos of TactileProcessor and ClassifyPhoneme.
#
# Instructions:
#
# 1. Install emscripten <https://emscripten.org>.
#
# 2. Build the js and wasm files by running `make` in this directory.
#
# 3. Open tactile_processor.html in a browser. Note that unfortunately
#    emscripten on Chrome is unable to load local files. The easiest workaround
#    is start a server from this directory with
#      python3 -m http.server 8080
#    Then open http://localhost:8080/tactile_processor.html

COMMON_OBJ= \
		basic_sdl_app.o \
		biquad_filter.o \
		carl_frontend_design.o \
		carl_frontend.o \
		complex.o \
		decibels.o \
		fast_fun.o \
		nn_ops.o \

TACTILE_PROCESSOR_DEMO_OBJ= \
		tactile_processor_web_bindings.o \
		butterworth.o \
		embed_vowel.o \
		enveloper.o \
		hexagon_interpolation.o \
		post_processor.o \
		run_tactile_processor_bracelet_assets.o \
		run_tactile_processor_sleeve_assets.o \
		tactile_processor.o \
		tactor_equalizer.o \
		texture_from_rle_data.o \
		$(COMMON_OBJ)

CLASSIFY_PHONEME_DEMO_OBJ= \
		classify_phoneme_web_bindings.o \
		classify_phoneme.o \
		$(COMMON_OBJ)

EMCC_FLAGS=-I ../../ -I ../../src -s USE_SDL=2 \
					 -s DISABLE_DEPRECATED_FIND_EVENT_TARGET_BEHAVIOR=1 -s ENVIRONMENT=web
EMCC_FLAGS+=-O3 -funsafe-math-optimizations

.PHONY: all clean rebuild
.SUFFIXES: .cpp .c .o

tactile_processor_web_bindings.js: $(TACTILE_PROCESSOR_DEMO_OBJ)
	emcc $(EMCC_FLAGS) $(TACTILE_PROCESSOR_DEMO_OBJ) -o $@

classify_phoneme_web_bindings.js: $(CLASSIFY_PHONEME_DEMO_OBJ)
	emcc $(EMCC_FLAGS) $(CLASSIFY_PHONEME_DEMO_OBJ) -o $@


tactile_processor_web_bindings.o: tactile_processor_web_bindings.cpp
	emcc $(EMCC_FLAGS) -std=c++17 -fno-rtti -fno-exceptions -c $< -o $@

classify_phoneme_web_bindings.o: classify_phoneme_web_bindings.cpp
	emcc $(EMCC_FLAGS) -std=c++17 -fno-rtti -fno-exceptions -c $< -o $@

biquad_filter.o: ../../src/dsp/biquad_filter.c
	emcc $(EMCC_FLAGS) -c $< -o $@

butterworth.o: ../../src/dsp/butterworth.c
	emcc $(EMCC_FLAGS) -c $< -o $@

complex.o: ../../src/dsp/complex.c
	emcc $(EMCC_FLAGS) -c $< -o $@

decibels.o: ../../src/dsp/decibels.c
	emcc $(EMCC_FLAGS) -c $< -o $@

fast_fun.o: ../../src/dsp/fast_fun.c
	emcc $(EMCC_FLAGS) -c $< -o $@

hexagon_interpolation.o: ../../src/phonetics/hexagon_interpolation.c
	emcc $(EMCC_FLAGS) -c $< -o $@

nn_ops.o: ../../src/phonetics/nn_ops.c
	emcc $(EMCC_FLAGS) -c $< -o $@

run_tactile_processor_bracelet_assets.o: ../tools/run_tactile_processor_bracelet_assets.c
	emcc $(EMCC_FLAGS) -c $< -o $@

run_tactile_processor_sleeve_assets.o: ../tools/run_tactile_processor_sleeve_assets.c
	emcc $(EMCC_FLAGS) -c $< -o $@

tactile_processor.o: ../../src/tactile/tactile_processor.c
	emcc $(EMCC_FLAGS) -c $< -o $@

post_processor.o: ../../src/tactile/post_processor.c
	emcc $(EMCC_FLAGS) -c $< -o $@

tactor_equalizer.o: ../../src/tactile/tactor_equalizer.c
	emcc $(EMCC_FLAGS) -c $< -o $@

enveloper.o: ../../src/tactile/enveloper.c
	emcc $(EMCC_FLAGS) -c $< -o $@

carl_frontend_design.o: ../../src/frontend/carl_frontend_design.c
	emcc $(EMCC_FLAGS) -c $< -o $@

carl_frontend.o: ../../src/frontend/carl_frontend.c
	emcc $(EMCC_FLAGS) -c $< -o $@

classify_phoneme.o: ../../src/phonetics/classify_phoneme.c
	emcc $(EMCC_FLAGS) -c $< -o $@

embed_vowel.o: ../../src/phonetics/embed_vowel.c
	emcc $(EMCC_FLAGS) -c $< -o $@

basic_sdl_app.o: ../tools/sdl/basic_sdl_app.c
	emcc $(EMCC_FLAGS) -c $< -o $@

texture_from_rle_data.o: ../tools/sdl/texture_from_rle_data.c
	emcc $(EMCC_FLAGS) -c $< -o $@

all: tactile_processor_web_bindings.js

clean:
	$(RM) tactile_processor_web_bindings.js tactile_processor_web_bindings.wasm \
			classify_phoneme_web_bindings.js classify_phoneme_web_bindings.wasm \
			*.o

rebuild: clean all
