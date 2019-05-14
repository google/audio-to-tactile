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

CFLAGS=-O2 -I. -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-sign-compare
LDFLAGS=-lm -lportaudio -lncurses -pthread

TACTOPHONE_OBJS=audio/tactile/phoneme_code/tactophone_main.o audio/tactile/phoneme_code/tactophone_state_main_menu.o audio/tactile/phoneme_code/tactophone_state_free_play.o audio/tactile/phoneme_code/tactophone_state_test_tactors.o audio/tactile/phoneme_code/tactophone_state_begin_lesson.o audio/tactile/phoneme_code/tactophone_state_lesson_trial.o audio/tactile/phoneme_code/tactophone_state_lesson_review.o audio/tactile/phoneme_code/tactophone_state_lesson_done.o audio/tactile/phoneme_code/phoneme_code.o audio/tactile/phoneme_code/tactophone_engine.o audio/tactile/phoneme_code/tactophone.o audio/tactile/phoneme_code/tactophone_lesson.o audio/tactile/util.o audio/tactile/tactile_player.o

PHONEME_CODE_TEST_OBJS=audio/tactile/phoneme_code/phoneme_code.o audio/tactile/util.o audio/tactile/phoneme_code/phoneme_code_test.o

TACTOPHONE_LESSON_TEST_OBJS=audio/tactile/phoneme_code/tactophone_lesson_test.o audio/tactile/phoneme_code/tactophone_lesson.o

TACTOPHONE_ENGINE_TEST_OBJS=audio/tactile/phoneme_code/tactophone_engine_test.o audio/tactile/phoneme_code/phoneme_code.o audio/tactile/phoneme_code/tactophone_lesson.o audio/tactile/phoneme_code/tactophone_engine.o audio/tactile/tactile_player.o audio/tactile/util.o

TACTILE_PLAYER_TEST_OBJS=audio/tactile/tactile_player_test.o audio/tactile/tactile_player.o

UTIL_TEST_OBJS=audio/tactile/util_test.o audio/tactile/util.o

ARCHIVE_NAME=tactophone_$(shell date -u +%Y%m%d)

.PHONY: clean default dist
.SUFFIXES: .c .o
default: tactophone

tactophone: $(TACTOPHONE_OBJS)
	$(CC) $(TACTOPHONE_OBJS) $(LDFLAGS) -o $@

phoneme_code_test: $(PHONEME_CODE_TEST_OBJS)
	$(CC) $(PHONEME_CODE_TEST_OBJS) $(LDFLAGS) -o $@

tactophone_engine_test: $(TACTOPHONE_ENGINE_TEST_OBJS)
	$(CC) $(TACTOPHONE_ENGINE_TEST_OBJS) $(LDFLAGS) -o $@

tactophone_lesson_test: $(TACTOPHONE_LESSON_TEST_OBJS)
	$(CC) $(TACTOPHONE_LESSON_TEST_OBJS) $(LDFLAGS) -o $@

tactile_player_test: $(TACTILE_PLAYER_TEST_OBJS)
	$(CC) $(TACTILE_PLAYER_TEST_OBJS) $(LDFLAGS) -o $@

util_test: $(UTIL_TEST_OBJS)
	$(CC) $(UTIL_TEST_OBJS) $(LDFLAGS) -o $@

.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

run_tests: phoneme_code_test tactophone_engine_test tactophone_lesson_test tactile_player_test util_test
	./phoneme_code_test
	./tactophone_engine_test
	./tactophone_lesson_test
	./tactile_player_test
	./util_test

clean:
	$(RM) $(TACTILE_PLAYER_TEST_OBJS) $(UTIL_TEST_OBJS) $(PHONEME_CODE_TEST_OBJS) $(TACTOPHONE_LESSON_TEST_OBJS) $(TACTOPHONE_ENGINE_TEST_OBJS) $(TACTOPHONE_OBJS) phoneme_code_test tactile_player_test util_test tactophone_lesson_test tactophone_engine_test tactophone

dist:
	-$(RM) -rf $(ARCHIVE_NAME)
	mkdir $(ARCHIVE_NAME)
	mkdir -p $(ARCHIVE_NAME)/doc
	mkdir -p $(ARCHIVE_NAME)/audio/dsp/portable
	mkdir -p $(ARCHIVE_NAME)/audio/tactile/phoneme_code
	ln README.md LICENSE CONTRIBUTING.md makefile lessons.txt $(ARCHIVE_NAME)/
	ln doc/* $(ARCHIVE_NAME)/doc
	ln audio/dsp/portable/*.c audio/dsp/portable/*.h $(ARCHIVE_NAME)/audio/dsp/portable/
	ln audio/tactile/phoneme_code/*.c audio/tactile/phoneme_code/*.h $(ARCHIVE_NAME)/audio/tactile/phoneme_code/
	ln audio/tactile/*.c audio/tactile/*.h $(ARCHIVE_NAME)/audio/tactile/
	tar vchzf $(ARCHIVE_NAME).tar.gz $(ARCHIVE_NAME)
	-$(RM) -rf $(ARCHIVE_NAME)
