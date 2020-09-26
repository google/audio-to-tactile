# Copyright 2020 Google LLC
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
# Makefile for tactile processor web demo.
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

DEMO_OBJ=biquad_filter.o \
				 butterworth.o \
				 complex.o \
				 fast_fun.o \
				 hexagon_interpolation.o \
				 nn_ops.o \
				 run_tactile_processor_assets.o \
				 tactile_processor.o \
				 energy_envelope.o \
				 carl_frontend_design.o \
				 carl_frontend.o \
				 embed_vowel.o \
				 basic_sdl_app.o \
				 texture_from_rle_data.o \
				 tactile_processor_web_bindings.o

EMCC_FLAGS=-I ../ -s USE_SDL=2 -s DISABLE_DEPRECATED_FIND_EVENT_TARGET_BEHAVIOR=1 -s ENVIRONMENT=web
EMCC_FLAGS+=-O3 -funsafe-math-optimizations

.PHONY: all clean rebuild
.SUFFIXES: .cc .c .o

tactile_processor_web_bindings.js: $(DEMO_OBJ)
	emcc $(EMCC_FLAGS) $(DEMO_OBJ) -o $@


tactile_processor_web_bindings.o: tactile_processor_web_bindings.cc
	emcc $(EMCC_FLAGS) -std=c++17 -fno-rtti -fno-exceptions -c $< -o $@

biquad_filter.o: ../dsp/biquad_filter.c
	emcc $(EMCC_FLAGS) -c $< -o $@

butterworth.o: ../dsp/butterworth.c
	emcc $(EMCC_FLAGS) -c $< -o $@

complex.o: ../dsp/complex.c
	emcc $(EMCC_FLAGS) -c $< -o $@

fast_fun.o: ../dsp/fast_fun.c
	emcc $(EMCC_FLAGS) -c $< -o $@

hexagon_interpolation.o: ../tactile/hexagon_interpolation.c
	emcc $(EMCC_FLAGS) -c $< -o $@

nn_ops.o: ../phonetics/nn_ops.c
	emcc $(EMCC_FLAGS) -c $< -o $@

run_tactile_processor_assets.o: ../tactile/run_tactile_processor_assets.c
	emcc $(EMCC_FLAGS) -c $< -o $@

tactile_processor.o: ../tactile/tactile_processor.c
	emcc $(EMCC_FLAGS) -c $< -o $@

energy_envelope.o: ../tactile/energy_envelope/energy_envelope.c
	emcc $(EMCC_FLAGS) -c $< -o $@

carl_frontend_design.o: ../frontend/carl_frontend_design.c
	emcc $(EMCC_FLAGS) -c $< -o $@

carl_frontend.o: ../frontend/carl_frontend.c
	emcc $(EMCC_FLAGS) -c $< -o $@

embed_vowel.o: ../phonetics/embed_vowel.c
	emcc $(EMCC_FLAGS) -c $< -o $@

basic_sdl_app.o: ../sdl/basic_sdl_app.c
	emcc $(EMCC_FLAGS) -c $< -o $@

texture_from_rle_data.o: ../sdl/texture_from_rle_data.c
	emcc $(EMCC_FLAGS) -c $< -o $@

all: tactile_processor_web_bindings.js

clean:
	$(RM) tactile_processor_web_bindings.js tactile_processor_web_bindings.wasm *.o

rebuild: clean all

