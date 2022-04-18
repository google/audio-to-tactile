/*
 * Copyright 2021 Google LLC
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     https://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This file contains a JSON formatted dictionary of available videos:
 * - Key is a unique reference for the stimulus group.
 * - 'answer' is the expected answer for the spoken content.
 * - 'trainVideos' is a list of filenames for the videos that should be used
 *   in the 'training' and 'practice' modes.
 * - 'testVideos' is a list of filenames for the videos that should be used
 *   in the 'test' mode.
 * - 'preTestVideos' is a list of filenames for the videos that should be used
 *   in the 'test' mode when the configuration indicates the task is a pretest.
 *
 * To add more videos, add an entry to the list, and include
 * the videos with the same filenames in the appropriate media
 * folder (defined as basePath in the main file).
 */

const wordsAndVideos = {
  'dhoo': {
    'answer': 'dh',
    'preTestVideos': ['CVs/F1/dhoo_1.mp4', 'CVs/M2/dhoo_1.mp4'],
    'trainVideos': ['CVs/F1/dhoo_2.mp4', 'CVs/M2/dhoo_2.mp4'],
    'testVideos': ['CVs/F1/dhoo_3.mp4', 'CVs/M2/dhoo_3.mp4'],
  },
  'yay': {
    'answer': 'y',
    'preTestVideos': ['CVs/F1/yay_1.mp4', 'CVs/M2/yay_1.mp4'],
    'trainVideos': ['CVs/F1/yay_2.mp4', 'CVs/M2/yay_2.mp4'],
    'testVideos': ['CVs/F1/yay_3.mp4', 'CVs/M2/yay_3.mp4'],
  },
  'koo': {
    'answer': 'k',
    'preTestVideos': ['CVs/F1/koo_1.mp4', 'CVs/M2/koo_1.mp4'],
    'trainVideos': ['CVs/F1/koo_2.mp4', 'CVs/M2/koo_2.mp4'],
    'testVideos': ['CVs/F1/koo_3.mp4', 'CVs/M2/koo_3.mp4'],
  },
  'woo': {
    'answer': 'w',
    'preTestVideos': ['CVs/F1/woo_1.mp4', 'CVs/M2/woo_1.mp4'],
    'trainVideos': ['CVs/F1/woo_2.mp4', 'CVs/M2/woo_2.mp4'],
    'testVideos': ['CVs/F1/woo_3.mp4', 'CVs/M2/woo_3.mp4'],
  },
  'shay': {
    'answer': 'sh',
    'preTestVideos': ['CVs/F1/shay_1.mp4', 'CVs/M2/shay_1.mp4'],
    'trainVideos': ['CVs/F1/shay_2.mp4', 'CVs/M2/shay_2.mp4'],
    'testVideos': ['CVs/F1/shay_3.mp4', 'CVs/M2/shay_3.mp4'],
  },
  'tah': {
    'answer': 't',
    'preTestVideos': ['CVs/F1/tah_1.mp4', 'CVs/M2/tah_1.mp4'],
    'trainVideos': ['CVs/F1/tah_2.mp4', 'CVs/M2/tah_2.mp4'],
    'testVideos': ['CVs/F1/tah_3.mp4', 'CVs/M2/tah_3.mp4'],
  },
  'shee': {
    'answer': 'sh',
    'preTestVideos': ['CVs/F1/shee_1.mp4', 'CVs/M2/shee_1.mp4'],
    'trainVideos': ['CVs/F1/shee_2.mp4', 'CVs/M2/shee_2.mp4'],
    'testVideos': ['CVs/F1/shee_3.mp4', 'CVs/M2/shee_3.mp4'],
  },
  'boo': {
    'answer': 'b',
    'preTestVideos': ['CVs/F1/boo_1.mp4', 'CVs/M2/boo_1.mp4'],
    'trainVideos': ['CVs/F1/boo_2.mp4', 'CVs/M2/boo_2.mp4'],
    'testVideos': ['CVs/F1/boo_3.mp4', 'CVs/M2/boo_3.mp4'],
  },
  'pay': {
    'answer': 'p',
    'preTestVideos': ['CVs/F1/pay_1.mp4', 'CVs/M2/pay_1.mp4'],
    'trainVideos': ['CVs/F1/pay_2.mp4', 'CVs/M2/pay_2.mp4'],
    'testVideos': ['CVs/F1/pay_3.mp4', 'CVs/M2/pay_3.mp4'],
  },
  'lay': {
    'answer': 'l',
    'preTestVideos': ['CVs/F1/lay_1.mp4', 'CVs/M2/lay_1.mp4'],
    'trainVideos': ['CVs/F1/lay_2.mp4', 'CVs/M2/lay_2.mp4'],
    'testVideos': ['CVs/F1/lay_3.mp4', 'CVs/M2/lay_3.mp4'],
  },
  'pee': {
    'answer': 'p',
    'preTestVideos': ['CVs/F1/pee_1.mp4', 'CVs/M2/pee_1.mp4'],
    'trainVideos': ['CVs/F1/pee_2.mp4', 'CVs/M2/pee_2.mp4'],
    'testVideos': ['CVs/F1/pee_3.mp4', 'CVs/M2/pee_3.mp4'],
  },
  'lee': {
    'answer': 'l',
    'preTestVideos': ['CVs/F1/lee_1.mp4', 'CVs/M2/lee_1.mp4'],
    'trainVideos': ['CVs/F1/lee_2.mp4', 'CVs/M2/lee_2.mp4'],
    'testVideos': ['CVs/F1/lee_3.mp4', 'CVs/M2/lee_3.mp4'],
  },
  'yee': {
    'answer': 'y',
    'preTestVideos': ['CVs/F1/yee_1.mp4', 'CVs/M2/yee_1.mp4'],
    'trainVideos': ['CVs/F1/yee_2.mp4', 'CVs/M2/yee_2.mp4'],
    'testVideos': ['CVs/F1/yee_3.mp4', 'CVs/M2/yee_3.mp4'],
  },
  'choh': {
    'answer': 'ch',
    'preTestVideos': ['CVs/F1/choh_1.mp4', 'CVs/M2/choh_1.mp4'],
    'trainVideos': ['CVs/F1/choh_2.mp4', 'CVs/M2/choh_2.mp4'],
    'testVideos': ['CVs/F1/choh_3.mp4', 'CVs/M2/choh_3.mp4'],
  },
  'joo': {
    'answer': 'j',
    'preTestVideos': ['CVs/F1/joo_1.mp4', 'CVs/M2/joo_1.mp4'],
    'trainVideos': ['CVs/F1/joo_2.mp4', 'CVs/M2/joo_2.mp4'],
    'testVideos': ['CVs/F1/joo_3.mp4', 'CVs/M2/joo_3.mp4'],
  },
  'chah': {
    'answer': 'ch',
    'preTestVideos': ['CVs/F1/chah_1.mp4', 'CVs/M2/chah_1.mp4'],
    'trainVideos': ['CVs/F1/chah_2.mp4', 'CVs/M2/chah_2.mp4'],
    'testVideos': ['CVs/F1/chah_3.mp4', 'CVs/M2/chah_3.mp4'],
  },
  'voo': {
    'answer': 'v',
    'preTestVideos': ['CVs/F1/voo_1.mp4', 'CVs/M2/voo_1.mp4'],
    'trainVideos': ['CVs/F1/voo_2.mp4', 'CVs/M2/voo_2.mp4'],
    'testVideos': ['CVs/F1/voo_3.mp4', 'CVs/M2/voo_3.mp4'],
  },
  'day': {
    'answer': 'd',
    'preTestVideos': ['CVs/F1/day_1.mp4', 'CVs/M2/day_1.mp4'],
    'trainVideos': ['CVs/F1/day_2.mp4', 'CVs/M2/day_2.mp4'],
    'testVideos': ['CVs/F1/day_3.mp4', 'CVs/M2/day_3.mp4'],
  },
  'mee': {
    'answer': 'm',
    'preTestVideos': ['CVs/F1/mee_1.mp4', 'CVs/M2/mee_1.mp4'],
    'trainVideos': ['CVs/F1/mee_2.mp4', 'CVs/M2/mee_2.mp4'],
    'testVideos': ['CVs/F1/mee_3.mp4', 'CVs/M2/mee_3.mp4'],
  },
  'toh': {
    'answer': 't',
    'preTestVideos': ['CVs/F1/toh_1.mp4', 'CVs/M2/toh_1.mp4'],
    'trainVideos': ['CVs/F1/toh_2.mp4', 'CVs/M2/toh_2.mp4'],
    'testVideos': ['CVs/F1/toh_3.mp4', 'CVs/M2/toh_3.mp4'],
  },
  'zhoo': {
    'answer': 'zh',
    'preTestVideos': ['CVs/F1/zhoo_1.mp4', 'CVs/M2/zhoo_1.mp4'],
    'trainVideos': ['CVs/F1/zhoo_2.mp4', 'CVs/M2/zhoo_2.mp4'],
    'testVideos': ['CVs/F1/zhoo_3.mp4', 'CVs/M2/zhoo_3.mp4'],
  },
  'may': {
    'answer': 'm',
    'preTestVideos': ['CVs/F1/may_1.mp4', 'CVs/M2/may_1.mp4'],
    'trainVideos': ['CVs/F1/may_2.mp4', 'CVs/M2/may_2.mp4'],
    'testVideos': ['CVs/F1/may_3.mp4', 'CVs/M2/may_3.mp4'],
  },
  'dee': {
    'answer': 'd',
    'preTestVideos': ['CVs/F1/dee_1.mp4', 'CVs/M2/dee_1.mp4'],
    'trainVideos': ['CVs/F1/dee_2.mp4', 'CVs/M2/dee_2.mp4'],
    'testVideos': ['CVs/F1/dee_3.mp4', 'CVs/M2/dee_3.mp4'],
  },
  'joh': {
    'answer': 'j',
    'preTestVideos': ['CVs/F1/joh_1.mp4', 'CVs/M2/joh_1.mp4'],
    'trainVideos': ['CVs/F1/joh_2.mp4', 'CVs/M2/joh_2.mp4'],
    'testVideos': ['CVs/F1/joh_3.mp4', 'CVs/M2/joh_3.mp4'],
  },
  'voh': {
    'answer': 'v',
    'preTestVideos': ['CVs/F1/voh_1.mp4', 'CVs/M2/voh_1.mp4'],
    'trainVideos': ['CVs/F1/voh_2.mp4', 'CVs/M2/voh_2.mp4'],
    'testVideos': ['CVs/F1/voh_3.mp4', 'CVs/M2/voh_3.mp4'],
  },
  'say': {
    'answer': 's',
    'preTestVideos': ['CVs/F1/say_1.mp4', 'CVs/M2/say_1.mp4'],
    'trainVideos': ['CVs/F1/say_2.mp4', 'CVs/M2/say_2.mp4'],
    'testVideos': ['CVs/F1/say_3.mp4', 'CVs/M2/say_3.mp4'],
  },
  'kah': {
    'answer': 'k',
    'preTestVideos': ['CVs/F1/kah_1.mp4', 'CVs/M2/kah_1.mp4'],
    'trainVideos': ['CVs/F1/kah_2.mp4', 'CVs/M2/kah_2.mp4'],
    'testVideos': ['CVs/F1/kah_3.mp4', 'CVs/M2/kah_3.mp4'],
  },
  'dhah': {
    'answer': 'dh',
    'preTestVideos': ['CVs/F1/dhah_1.mp4', 'CVs/M2/dhah_1.mp4'],
    'trainVideos': ['CVs/F1/dhah_2.mp4', 'CVs/M2/dhah_2.mp4'],
    'testVideos': ['CVs/F1/dhah_3.mp4', 'CVs/M2/dhah_3.mp4'],
  },
  'wah': {
    'answer': 'w',
    'preTestVideos': ['CVs/F1/wah_1.mp4', 'CVs/M2/wah_1.mp4'],
    'trainVideos': ['CVs/F1/wah_2.mp4', 'CVs/M2/wah_2.mp4'],
    'testVideos': ['CVs/F1/wah_3.mp4', 'CVs/M2/wah_3.mp4'],
  },
  'zee': {
    'answer': 'z',
    'preTestVideos': ['CVs/F1/zee_1.mp4', 'CVs/M2/zee_1.mp4'],
    'trainVideos': ['CVs/F1/zee_2.mp4', 'CVs/M2/zee_2.mp4'],
    'testVideos': ['CVs/F1/zee_3.mp4', 'CVs/M2/zee_3.mp4'],
  },
  'too': {
    'answer': 't',
    'preTestVideos': ['CVs/F1/too_1.mp4', 'CVs/M2/too_1.mp4'],
    'trainVideos': ['CVs/F1/too_2.mp4', 'CVs/M2/too_2.mp4'],
    'testVideos': ['CVs/F1/too_3.mp4', 'CVs/M2/too_3.mp4'],
  },
  'zhoh': {
    'answer': 'zh',
    'preTestVideos': ['CVs/F1/zhoh_1.mp4', 'CVs/M2/zhoh_1.mp4'],
    'trainVideos': ['CVs/F1/zhoh_2.mp4', 'CVs/M2/zhoh_2.mp4'],
    'testVideos': ['CVs/F1/zhoh_3.mp4', 'CVs/M2/zhoh_3.mp4'],
  },
  'fay': {
    'answer': 'f',
    'preTestVideos': ['CVs/F1/fay_1.mp4', 'CVs/M2/fay_1.mp4'],
    'trainVideos': ['CVs/F1/fay_2.mp4', 'CVs/M2/fay_2.mp4'],
    'testVideos': ['CVs/F1/fay_3.mp4', 'CVs/M2/fay_3.mp4'],
  },
  'zay': {
    'answer': 'z',
    'preTestVideos': ['CVs/F1/zay_1.mp4', 'CVs/M2/zay_1.mp4'],
    'trainVideos': ['CVs/F1/zay_2.mp4', 'CVs/M2/zay_2.mp4'],
    'testVideos': ['CVs/F1/zay_3.mp4', 'CVs/M2/zay_3.mp4'],
  },
  'bah': {
    'answer': 'b',
    'preTestVideos': ['CVs/F1/bah_1.mp4', 'CVs/M2/bah_1.mp4'],
    'trainVideos': ['CVs/F1/bah_2.mp4', 'CVs/M2/bah_2.mp4'],
    'testVideos': ['CVs/F1/bah_3.mp4', 'CVs/M2/bah_3.mp4'],
  },
  'fee': {
    'answer': 'f',
    'preTestVideos': ['CVs/F1/fee_1.mp4', 'CVs/M2/fee_1.mp4'],
    'trainVideos': ['CVs/F1/fee_2.mp4', 'CVs/M2/fee_2.mp4'],
    'testVideos': ['CVs/F1/fee_3.mp4', 'CVs/M2/fee_3.mp4'],
  },
  'see': {
    'answer': 's',
    'preTestVideos': ['CVs/F1/see_1.mp4', 'CVs/M2/see_1.mp4'],
    'trainVideos': ['CVs/F1/see_2.mp4', 'CVs/M2/see_2.mp4'],
    'testVideos': ['CVs/F1/see_3.mp4', 'CVs/M2/see_3.mp4'],
  },
  'choo': {
    'answer': 'ch',
    'preTestVideos': ['CVs/F1/choo_1.mp4', 'CVs/M2/choo_1.mp4'],
    'trainVideos': ['CVs/F1/choo_2.mp4', 'CVs/M2/choo_2.mp4'],
    'testVideos': ['CVs/F1/choo_3.mp4', 'CVs/M2/choo_3.mp4'],
  },
  'jah': {
    'answer': 'j',
    'preTestVideos': ['CVs/F1/jah_1.mp4', 'CVs/M2/jah_1.mp4'],
    'trainVideos': ['CVs/F1/jah_2.mp4', 'CVs/M2/jah_2.mp4'],
    'testVideos': ['CVs/F1/jah_3.mp4', 'CVs/M2/jah_3.mp4'],
  },
  'vah': {
    'answer': 'v',
    'preTestVideos': ['CVs/F1/vah_1.mp4', 'CVs/M2/vah_1.mp4'],
    'trainVideos': ['CVs/F1/vah_2.mp4', 'CVs/M2/vah_2.mp4'],
    'testVideos': ['CVs/F1/vah_3.mp4', 'CVs/M2/vah_3.mp4'],
  },
  'koh': {
    'answer': 'k',
    'preTestVideos': ['CVs/F1/koh_1.mp4', 'CVs/M2/koh_1.mp4'],
    'trainVideos': ['CVs/F1/koh_2.mp4', 'CVs/M2/koh_2.mp4'],
    'testVideos': ['CVs/F1/koh_3.mp4', 'CVs/M2/koh_3.mp4'],
  },
  'nay': {
    'answer': 'n',
    'preTestVideos': ['CVs/F1/nay_1.mp4', 'CVs/M2/nay_1.mp4'],
    'trainVideos': ['CVs/F1/nay_2.mp4', 'CVs/M2/nay_2.mp4'],
    'testVideos': ['CVs/F1/nay_3.mp4', 'CVs/M2/nay_3.mp4'],
  },
  'ree': {
    'answer': 'r',
    'preTestVideos': ['CVs/F1/ree_1.mp4', 'CVs/M2/ree_1.mp4'],
    'trainVideos': ['CVs/F1/ree_2.mp4', 'CVs/M2/ree_2.mp4'],
    'testVideos': ['CVs/F1/ree_3.mp4', 'CVs/M2/ree_3.mp4'],
  },
  'ray': {
    'answer': 'r',
    'preTestVideos': ['CVs/F1/ray_1.mp4', 'CVs/M2/ray_1.mp4'],
    'trainVideos': ['CVs/F1/ray_2.mp4', 'CVs/M2/ray_2.mp4'],
    'testVideos': ['CVs/F1/ray_3.mp4', 'CVs/M2/ray_3.mp4'],
  },
  'woh': {
    'answer': 'w',
    'preTestVideos': ['CVs/F1/woh_1.mp4', 'CVs/M2/woh_1.mp4'],
    'trainVideos': ['CVs/F1/woh_2.mp4', 'CVs/M2/woh_2.mp4'],
    'testVideos': ['CVs/F1/woh_3.mp4', 'CVs/M2/woh_3.mp4'],
  },
  'zhah': {
    'answer': 'zh',
    'preTestVideos': ['CVs/F1/zhah_1.mp4', 'CVs/M2/zhah_1.mp4'],
    'trainVideos': ['CVs/F1/zhah_2.mp4', 'CVs/M2/zhah_2.mp4'],
    'testVideos': ['CVs/F1/zhah_3.mp4', 'CVs/M2/zhah_3.mp4'],
  },
  'thay': {
    'answer': 'th',
    'preTestVideos': ['CVs/F1/thay_1.mp4', 'CVs/M2/thay_1.mp4'],
    'trainVideos': ['CVs/F1/thay_2.mp4', 'CVs/M2/thay_2.mp4'],
    'testVideos': ['CVs/F1/thay_3.mp4', 'CVs/M2/thay_3.mp4'],
  },
  'boh': {
    'answer': 'b',
    'preTestVideos': ['CVs/F1/boh_1.mp4', 'CVs/M2/boh_1.mp4'],
    'trainVideos': ['CVs/F1/boh_2.mp4', 'CVs/M2/boh_2.mp4'],
    'testVideos': ['CVs/F1/boh_3.mp4', 'CVs/M2/boh_3.mp4'],
  },
  'gay': {
    'answer': 'g',
    'preTestVideos': ['CVs/F1/gay_1.mp4', 'CVs/M2/gay_1.mp4'],
    'trainVideos': ['CVs/F1/gay_2.mp4', 'CVs/M2/gay_2.mp4'],
    'testVideos': ['CVs/F1/gay_3.mp4', 'CVs/M2/gay_3.mp4'],
  },
  'gee': {
    'answer': 'g',
    'preTestVideos': ['CVs/F1/gee_1.mp4', 'CVs/M2/gee_1.mp4'],
    'trainVideos': ['CVs/F1/gee_2.mp4', 'CVs/M2/gee_2.mp4'],
    'testVideos': ['CVs/F1/gee_3.mp4', 'CVs/M2/gee_3.mp4'],
  },
  'thee': {
    'answer': 'th',
    'preTestVideos': ['CVs/F1/thee_1.mp4', 'CVs/M2/thee_1.mp4'],
    'trainVideos': ['CVs/F1/thee_2.mp4', 'CVs/M2/thee_2.mp4'],
    'testVideos': ['CVs/F1/thee_3.mp4', 'CVs/M2/thee_3.mp4'],
  },
  'nee': {
    'answer': 'n',
    'preTestVideos': ['CVs/F1/nee_1.mp4', 'CVs/M2/nee_1.mp4'],
    'trainVideos': ['CVs/F1/nee_2.mp4', 'CVs/M2/nee_2.mp4'],
    'testVideos': ['CVs/F1/nee_3.mp4', 'CVs/M2/nee_3.mp4'],
  },
  'dhoh': {
    'answer': 'dh',
    'preTestVideos': ['CVs/F1/dhoh_1.mp4', 'CVs/M2/dhoh_1.mp4'],
    'trainVideos': ['CVs/F1/dhoh_2.mp4', 'CVs/M2/dhoh_2.mp4'],
    'testVideos': ['CVs/F1/dhoh_3.mp4', 'CVs/M2/dhoh_3.mp4'],
  },
  'thoh': {
    'answer': 'th',
    'preTestVideos': ['CVs/F1/thoh_1.mp4', 'CVs/M2/thoh_1.mp4'],
    'trainVideos': ['CVs/F1/thoh_2.mp4', 'CVs/M2/thoh_2.mp4'],
    'testVideos': ['CVs/F1/thoh_3.mp4', 'CVs/M2/thoh_3.mp4'],
  },
  'bay': {
    'answer': 'b',
    'preTestVideos': ['CVs/F1/bay_1.mp4', 'CVs/M2/bay_1.mp4'],
    'trainVideos': ['CVs/F1/bay_2.mp4', 'CVs/M2/bay_2.mp4'],
    'testVideos': ['CVs/F1/bay_3.mp4', 'CVs/M2/bay_3.mp4'],
  },
  'poo': {
    'answer': 'p',
    'preTestVideos': ['CVs/F1/poo_1.mp4', 'CVs/M2/poo_1.mp4'],
    'trainVideos': ['CVs/F1/poo_2.mp4', 'CVs/M2/poo_2.mp4'],
    'testVideos': ['CVs/F1/poo_3.mp4', 'CVs/M2/poo_3.mp4'],
  },
  'bee': {
    'answer': 'b',
    'preTestVideos': ['CVs/F1/bee_1.mp4', 'CVs/M2/bee_1.mp4'],
    'trainVideos': ['CVs/F1/bee_2.mp4', 'CVs/M2/bee_2.mp4'],
    'testVideos': ['CVs/F1/bee_3.mp4', 'CVs/M2/bee_3.mp4'],
  },
  'shoo': {
    'answer': 'sh',
    'preTestVideos': ['CVs/F1/shoo_1.mp4', 'CVs/M2/shoo_1.mp4'],
    'trainVideos': ['CVs/F1/shoo_2.mp4', 'CVs/M2/shoo_2.mp4'],
    'testVideos': ['CVs/F1/shoo_3.mp4', 'CVs/M2/shoo_3.mp4'],
  },
  'zah': {
    'answer': 'z',
    'preTestVideos': ['CVs/F1/zah_1.mp4', 'CVs/M2/zah_1.mp4'],
    'trainVideos': ['CVs/F1/zah_2.mp4', 'CVs/M2/zah_2.mp4'],
    'testVideos': ['CVs/F1/zah_3.mp4', 'CVs/M2/zah_3.mp4'],
  },
  'wee': {
    'answer': 'w',
    'preTestVideos': ['CVs/F1/wee_1.mp4', 'CVs/M2/wee_1.mp4'],
    'trainVideos': ['CVs/F1/wee_2.mp4', 'CVs/M2/wee_2.mp4'],
    'testVideos': ['CVs/F1/wee_3.mp4', 'CVs/M2/wee_3.mp4'],
  },
  'noh': {
    'answer': 'n',
    'preTestVideos': ['CVs/F1/noh_1.mp4', 'CVs/M2/noh_1.mp4'],
    'trainVideos': ['CVs/F1/noh_2.mp4', 'CVs/M2/noh_2.mp4'],
    'testVideos': ['CVs/F1/noh_3.mp4', 'CVs/M2/noh_3.mp4'],
  },
  'dhay': {
    'answer': 'dh',
    'preTestVideos': ['CVs/F1/dhay_1.mp4', 'CVs/M2/dhay_1.mp4'],
    'trainVideos': ['CVs/F1/dhay_2.mp4', 'CVs/M2/dhay_2.mp4'],
    'testVideos': ['CVs/F1/dhay_3.mp4', 'CVs/M2/dhay_3.mp4'],
  },
  'dhee': {
    'answer': 'dh',
    'preTestVideos': ['CVs/F1/dhee_1.mp4', 'CVs/M2/dhee_1.mp4'],
    'trainVideos': ['CVs/F1/dhee_2.mp4', 'CVs/M2/dhee_2.mp4'],
    'testVideos': ['CVs/F1/dhee_3.mp4', 'CVs/M2/dhee_3.mp4'],
  },
  'way': {
    'answer': 'w',
    'preTestVideos': ['CVs/F1/way_1.mp4', 'CVs/M2/way_1.mp4'],
    'trainVideos': ['CVs/F1/way_2.mp4', 'CVs/M2/way_2.mp4'],
    'testVideos': ['CVs/F1/way_3.mp4', 'CVs/M2/way_3.mp4'],
  },
  'kee': {
    'answer': 'k',
    'preTestVideos': ['CVs/F1/kee_1.mp4', 'CVs/M2/kee_1.mp4'],
    'trainVideos': ['CVs/F1/kee_2.mp4', 'CVs/M2/kee_2.mp4'],
    'testVideos': ['CVs/F1/kee_3.mp4', 'CVs/M2/kee_3.mp4'],
  },
  'sah': {
    'answer': 's',
    'preTestVideos': ['CVs/F1/sah_1.mp4', 'CVs/M2/sah_1.mp4'],
    'trainVideos': ['CVs/F1/sah_2.mp4', 'CVs/M2/sah_2.mp4'],
    'testVideos': ['CVs/F1/sah_3.mp4', 'CVs/M2/sah_3.mp4'],
  },
  'roh': {
    'answer': 'r',
    'preTestVideos': ['CVs/F1/roh_1.mp4', 'CVs/M2/roh_1.mp4'],
    'trainVideos': ['CVs/F1/roh_2.mp4', 'CVs/M2/roh_2.mp4'],
    'testVideos': ['CVs/F1/roh_3.mp4', 'CVs/M2/roh_3.mp4'],
  },
  'yoo': {
    'answer': 'y',
    'preTestVideos': ['CVs/F1/yoo_1.mp4', 'CVs/M2/yoo_1.mp4'],
    'trainVideos': ['CVs/F1/yoo_2.mp4', 'CVs/M2/yoo_2.mp4'],
    'testVideos': ['CVs/F1/yoo_3.mp4', 'CVs/M2/yoo_3.mp4'],
  },
  'kay': {
    'answer': 'k',
    'preTestVideos': ['CVs/F1/kay_1.mp4', 'CVs/M2/kay_1.mp4'],
    'trainVideos': ['CVs/F1/kay_2.mp4', 'CVs/M2/kay_2.mp4'],
    'testVideos': ['CVs/F1/kay_3.mp4', 'CVs/M2/kay_3.mp4'],
  },
  'fah': {
    'answer': 'f',
    'preTestVideos': ['CVs/F1/fah_1.mp4', 'CVs/M2/fah_1.mp4'],
    'trainVideos': ['CVs/F1/fah_2.mp4', 'CVs/M2/fah_2.mp4'],
    'testVideos': ['CVs/F1/fah_3.mp4', 'CVs/M2/fah_3.mp4'],
  },
  'loo': {
    'answer': 'l',
    'preTestVideos': ['CVs/F1/loo_1.mp4', 'CVs/M2/loo_1.mp4'],
    'trainVideos': ['CVs/F1/loo_2.mp4', 'CVs/M2/loo_2.mp4'],
    'testVideos': ['CVs/F1/loo_3.mp4', 'CVs/M2/loo_3.mp4'],
  },
  'goh': {
    'answer': 'g',
    'preTestVideos': ['CVs/F1/goh_1.mp4', 'CVs/M2/goh_1.mp4'],
    'trainVideos': ['CVs/F1/goh_2.mp4', 'CVs/M2/goh_2.mp4'],
    'testVideos': ['CVs/F1/goh_3.mp4', 'CVs/M2/goh_3.mp4'],
  },
  'thah': {
    'answer': 'th',
    'preTestVideos': ['CVs/F1/thah_1.mp4', 'CVs/M2/thah_1.mp4'],
    'trainVideos': ['CVs/F1/thah_2.mp4', 'CVs/M2/thah_2.mp4'],
    'testVideos': ['CVs/F1/thah_3.mp4', 'CVs/M2/thah_3.mp4'],
  },
  'gah': {
    'answer': 'g',
    'preTestVideos': ['CVs/F1/gah_1.mp4', 'CVs/M2/gah_1.mp4'],
    'trainVideos': ['CVs/F1/gah_2.mp4', 'CVs/M2/gah_2.mp4'],
    'testVideos': ['CVs/F1/gah_3.mp4', 'CVs/M2/gah_3.mp4'],
  },
  'foh': {
    'answer': 'f',
    'preTestVideos': ['CVs/F1/foh_1.mp4', 'CVs/M2/foh_1.mp4'],
    'trainVideos': ['CVs/F1/foh_2.mp4', 'CVs/M2/foh_2.mp4'],
    'testVideos': ['CVs/F1/foh_3.mp4', 'CVs/M2/foh_3.mp4'],
  },
  'zhay': {
    'answer': 'zh',
    'preTestVideos': ['CVs/F1/zhay_1.mp4', 'CVs/M2/zhay_1.mp4'],
    'trainVideos': ['CVs/F1/zhay_2.mp4', 'CVs/M2/zhay_2.mp4'],
    'testVideos': ['CVs/F1/zhay_3.mp4', 'CVs/M2/zhay_3.mp4'],
  },
  'moo': {
    'answer': 'm',
    'preTestVideos': ['CVs/F1/moo_1.mp4', 'CVs/M2/moo_1.mp4'],
    'trainVideos': ['CVs/F1/moo_2.mp4', 'CVs/M2/moo_2.mp4'],
    'testVideos': ['CVs/F1/moo_3.mp4', 'CVs/M2/moo_3.mp4'],
  },
  'zhee': {
    'answer': 'zh',
    'preTestVideos': ['CVs/F1/zhee_1.mp4', 'CVs/M2/zhee_1.mp4'],
    'trainVideos': ['CVs/F1/zhee_2.mp4', 'CVs/M2/zhee_2.mp4'],
    'testVideos': ['CVs/F1/zhee_3.mp4', 'CVs/M2/zhee_3.mp4'],
  },
  'rah': {
    'answer': 'r',
    'preTestVideos': ['CVs/F1/rah_1.mp4', 'CVs/M2/rah_1.mp4'],
    'trainVideos': ['CVs/F1/rah_2.mp4', 'CVs/M2/rah_2.mp4'],
    'testVideos': ['CVs/F1/rah_3.mp4', 'CVs/M2/rah_3.mp4'],
  },
  'vee': {
    'answer': 'v',
    'preTestVideos': ['CVs/F1/vee_1.mp4', 'CVs/M2/vee_1.mp4'],
    'trainVideos': ['CVs/F1/vee_2.mp4', 'CVs/M2/vee_2.mp4'],
    'testVideos': ['CVs/F1/vee_3.mp4', 'CVs/M2/vee_3.mp4'],
  },
  'jay': {
    'answer': 'j',
    'preTestVideos': ['CVs/F1/jay_1.mp4', 'CVs/M2/jay_1.mp4'],
    'trainVideos': ['CVs/F1/jay_2.mp4', 'CVs/M2/jay_2.mp4'],
    'testVideos': ['CVs/F1/jay_3.mp4', 'CVs/M2/jay_3.mp4'],
  },
  'jee': {
    'answer': 'j',
    'preTestVideos': ['CVs/F1/jee_1.mp4', 'CVs/M2/jee_1.mp4'],
    'trainVideos': ['CVs/F1/jee_2.mp4', 'CVs/M2/jee_2.mp4'],
    'testVideos': ['CVs/F1/jee_3.mp4', 'CVs/M2/jee_3.mp4'],
  },
  'soh': {
    'answer': 's',
    'preTestVideos': ['CVs/F1/soh_1.mp4', 'CVs/M2/soh_1.mp4'],
    'trainVideos': ['CVs/F1/soh_2.mp4', 'CVs/M2/soh_2.mp4'],
    'testVideos': ['CVs/F1/soh_3.mp4', 'CVs/M2/soh_3.mp4'],
  },
  'vay': {
    'answer': 'v',
    'preTestVideos': ['CVs/F1/vay_1.mp4', 'CVs/M2/vay_1.mp4'],
    'trainVideos': ['CVs/F1/vay_2.mp4', 'CVs/M2/vay_2.mp4'],
    'testVideos': ['CVs/F1/vay_3.mp4', 'CVs/M2/vay_3.mp4'],
  },
  'doo': {
    'answer': 'd',
    'preTestVideos': ['CVs/F1/doo_1.mp4', 'CVs/M2/doo_1.mp4'],
    'trainVideos': ['CVs/F1/doo_2.mp4', 'CVs/M2/doo_2.mp4'],
    'testVideos': ['CVs/F1/doo_3.mp4', 'CVs/M2/doo_3.mp4'],
  },
  'nah': {
    'answer': 'n',
    'preTestVideos': ['CVs/F1/nah_1.mp4', 'CVs/M2/nah_1.mp4'],
    'trainVideos': ['CVs/F1/nah_2.mp4', 'CVs/M2/nah_2.mp4'],
    'testVideos': ['CVs/F1/nah_3.mp4', 'CVs/M2/nah_3.mp4'],
  },
  'zoh': {
    'answer': 'z',
    'preTestVideos': ['CVs/F1/zoh_1.mp4', 'CVs/M2/zoh_1.mp4'],
    'trainVideos': ['CVs/F1/zoh_2.mp4', 'CVs/M2/zoh_2.mp4'],
    'testVideos': ['CVs/F1/zoh_3.mp4', 'CVs/M2/zoh_3.mp4'],
  },
  'pah': {
    'answer': 'p',
    'preTestVideos': ['CVs/F1/pah_1.mp4', 'CVs/M2/pah_1.mp4'],
    'trainVideos': ['CVs/F1/pah_2.mp4', 'CVs/M2/pah_2.mp4'],
    'testVideos': ['CVs/F1/pah_3.mp4', 'CVs/M2/pah_3.mp4'],
  },
  'shah': {
    'answer': 'sh',
    'preTestVideos': ['CVs/F1/shah_1.mp4', 'CVs/M2/shah_1.mp4'],
    'trainVideos': ['CVs/F1/shah_2.mp4', 'CVs/M2/shah_2.mp4'],
    'testVideos': ['CVs/F1/shah_3.mp4', 'CVs/M2/shah_3.mp4'],
  },
  'tee': {
    'answer': 't',
    'preTestVideos': ['CVs/F1/tee_1.mp4', 'CVs/M2/tee_1.mp4'],
    'trainVideos': ['CVs/F1/tee_2.mp4', 'CVs/M2/tee_2.mp4'],
    'testVideos': ['CVs/F1/tee_3.mp4', 'CVs/M2/tee_3.mp4'],
  },
  'zoo': {
    'answer': 'z',
    'preTestVideos': ['CVs/F1/zoo_1.mp4', 'CVs/M2/zoo_1.mp4'],
    'trainVideos': ['CVs/F1/zoo_2.mp4', 'CVs/M2/zoo_2.mp4'],
    'testVideos': ['CVs/F1/zoo_3.mp4', 'CVs/M2/zoo_3.mp4'],
  },
  'doh': {
    'answer': 'd',
    'preTestVideos': ['CVs/F1/doh_1.mp4', 'CVs/M2/doh_1.mp4'],
    'trainVideos': ['CVs/F1/doh_2.mp4', 'CVs/M2/doh_2.mp4'],
    'testVideos': ['CVs/F1/doh_3.mp4', 'CVs/M2/doh_3.mp4'],
  },
  'soo': {
    'answer': 's',
    'preTestVideos': ['CVs/F1/soo_1.mp4', 'CVs/M2/soo_1.mp4'],
    'trainVideos': ['CVs/F1/soo_2.mp4', 'CVs/M2/soo_2.mp4'],
    'testVideos': ['CVs/F1/soo_3.mp4', 'CVs/M2/soo_3.mp4'],
  },
  'yah': {
    'answer': 'y',
    'preTestVideos': ['CVs/F1/yah_1.mp4', 'CVs/M2/yah_1.mp4'],
    'trainVideos': ['CVs/F1/yah_2.mp4', 'CVs/M2/yah_2.mp4'],
    'testVideos': ['CVs/F1/yah_3.mp4', 'CVs/M2/yah_3.mp4'],
  },
  'moh': {
    'answer': 'm',
    'preTestVideos': ['CVs/F1/moh_1.mp4', 'CVs/M2/moh_1.mp4'],
    'trainVideos': ['CVs/F1/moh_2.mp4', 'CVs/M2/moh_2.mp4'],
    'testVideos': ['CVs/F1/moh_3.mp4', 'CVs/M2/moh_3.mp4'],
  },
  'tay': {
    'answer': 't',
    'preTestVideos': ['CVs/F1/tay_1.mp4', 'CVs/M2/tay_1.mp4'],
    'trainVideos': ['CVs/F1/tay_2.mp4', 'CVs/M2/tay_2.mp4'],
    'testVideos': ['CVs/F1/tay_3.mp4', 'CVs/M2/tay_3.mp4'],
  },
  'foo': {
    'answer': 'f',
    'preTestVideos': ['CVs/F1/foo_1.mp4', 'CVs/M2/foo_1.mp4'],
    'trainVideos': ['CVs/F1/foo_2.mp4', 'CVs/M2/foo_2.mp4'],
    'testVideos': ['CVs/F1/foo_3.mp4', 'CVs/M2/foo_3.mp4'],
  },
  'lah': {
    'answer': 'l',
    'preTestVideos': ['CVs/F1/lah_1.mp4', 'CVs/M2/lah_1.mp4'],
    'trainVideos': ['CVs/F1/lah_2.mp4', 'CVs/M2/lah_2.mp4'],
    'testVideos': ['CVs/F1/lah_3.mp4', 'CVs/M2/lah_3.mp4'],
  },
  'goo': {
    'answer': 'g',
    'preTestVideos': ['CVs/F1/goo_1.mp4', 'CVs/M2/goo_1.mp4'],
    'trainVideos': ['CVs/F1/goo_2.mp4', 'CVs/M2/goo_2.mp4'],
    'testVideos': ['CVs/F1/goo_3.mp4', 'CVs/M2/goo_3.mp4'],
  },
  'thoo': {
    'answer': 'th',
    'preTestVideos': ['CVs/F1/thoo_1.mp4', 'CVs/M2/thoo_1.mp4'],
    'trainVideos': ['CVs/F1/thoo_2.mp4', 'CVs/M2/thoo_2.mp4'],
    'testVideos': ['CVs/F1/thoo_3.mp4', 'CVs/M2/thoo_3.mp4'],
  },
  'loh': {
    'answer': 'l',
    'preTestVideos': ['CVs/F1/loh_1.mp4', 'CVs/M2/loh_1.mp4'],
    'trainVideos': ['CVs/F1/loh_2.mp4', 'CVs/M2/loh_2.mp4'],
    'testVideos': ['CVs/F1/loh_3.mp4', 'CVs/M2/loh_3.mp4'],
  },
  'shoh': {
    'answer': 'sh',
    'preTestVideos': ['CVs/F1/shoh_1.mp4', 'CVs/M2/shoh_1.mp4'],
    'trainVideos': ['CVs/F1/shoh_2.mp4', 'CVs/M2/shoh_2.mp4'],
    'testVideos': ['CVs/F1/shoh_3.mp4', 'CVs/M2/shoh_3.mp4'],
  },
  'mah': {
    'answer': 'm',
    'preTestVideos': ['CVs/F1/mah_1.mp4', 'CVs/M2/mah_1.mp4'],
    'trainVideos': ['CVs/F1/mah_2.mp4', 'CVs/M2/mah_2.mp4'],
    'testVideos': ['CVs/F1/mah_3.mp4', 'CVs/M2/mah_3.mp4'],
  },
  'yoh': {
    'answer': 'y',
    'preTestVideos': ['CVs/F1/yoh_1.mp4', 'CVs/M2/yoh_1.mp4'],
    'trainVideos': ['CVs/F1/yoh_2.mp4', 'CVs/M2/yoh_2.mp4'],
    'testVideos': ['CVs/F1/yoh_3.mp4', 'CVs/M2/yoh_3.mp4'],
  },
  'roo': {
    'answer': 'r',
    'preTestVideos': ['CVs/F1/roo_1.mp4', 'CVs/M2/roo_1.mp4'],
    'trainVideos': ['CVs/F1/roo_2.mp4', 'CVs/M2/roo_2.mp4'],
    'testVideos': ['CVs/F1/roo_3.mp4', 'CVs/M2/roo_3.mp4'],
  },
  'chay': {
    'answer': 'ch',
    'preTestVideos': ['CVs/F1/chay_1.mp4', 'CVs/M2/chay_1.mp4'],
    'trainVideos': ['CVs/F1/chay_2.mp4', 'CVs/M2/chay_2.mp4'],
    'testVideos': ['CVs/F1/chay_3.mp4', 'CVs/M2/chay_3.mp4'],
  },
  'chee': {
    'answer': 'ch',
    'preTestVideos': ['CVs/F1/chee_1.mp4', 'CVs/M2/chee_1.mp4'],
    'trainVideos': ['CVs/F1/chee_2.mp4', 'CVs/M2/chee_2.mp4'],
    'testVideos': ['CVs/F1/chee_3.mp4', 'CVs/M2/chee_3.mp4'],
  },
  'dah': {
    'answer': 'd',
    'preTestVideos': ['CVs/F1/dah_1.mp4', 'CVs/M2/dah_1.mp4'],
    'trainVideos': ['CVs/F1/dah_2.mp4', 'CVs/M2/dah_2.mp4'],
    'testVideos': ['CVs/F1/dah_3.mp4', 'CVs/M2/dah_3.mp4'],
  },
  'noo': {
    'answer': 'n',
    'preTestVideos': ['CVs/F1/noo_1.mp4', 'CVs/M2/noo_1.mp4'],
    'trainVideos': ['CVs/F1/noo_2.mp4', 'CVs/M2/noo_2.mp4'],
    'testVideos': ['CVs/F1/noo_3.mp4', 'CVs/M2/noo_3.mp4'],
  },
  'poh': {
    'answer': 'p',
    'preTestVideos': ['CVs/F1/poh_1.mp4', 'CVs/M2/poh_1.mp4'],
    'trainVideos': ['CVs/F1/poh_2.mp4', 'CVs/M2/poh_2.mp4'],
    'testVideos': ['CVs/F1/poh_3.mp4', 'CVs/M2/poh_3.mp4'],
  },
  'hawed': {
    'answer': 'hawed',
    'preTestVideos': ['hVds/F1/hawed_1.mp4', 'hVds/M2/hawed_1.mp4'],
    'trainVideos': ['hVds/F1/hawed_2.mp4', 'hVds/M2/hawed_2.mp4'],
    'testVideos': ['hVds/F1/hawed_3.mp4', 'hVds/M2/hawed_3.mp4'],
  },
  'head': {
    'answer': 'head',
    'preTestVideos': ['hVds/F1/head_1.mp4', 'hVds/M2/head_1.mp4'],
    'trainVideos': ['hVds/F1/head_2.mp4', 'hVds/M2/head_2.mp4'],
    'testVideos': ['hVds/F1/head_3.mp4', 'hVds/M2/head_3.mp4'],
  },
  'hoed': {
    'answer': 'hoed',
    'preTestVideos': ['hVds/F1/hoed_1.mp4', 'hVds/M2/hoed_1.mp4'],
    'trainVideos': ['hVds/F1/hoed_2.mp4', 'hVds/M2/hoed_2.mp4'],
    'testVideos': ['hVds/F1/hoed_3.mp4', 'hVds/M2/hoed_3.mp4'],
  },
  'heard': {
    'answer': 'heard',
    'preTestVideos': ['hVds/F1/heard_1.mp4', 'hVds/M2/heard_1.mp4'],
    'trainVideos': ['hVds/F1/heard_2.mp4', 'hVds/M2/heard_2.mp4'],
    'testVideos': ['hVds/F1/heard_3.mp4', 'hVds/M2/heard_3.mp4'],
  },
  'hoyed': {
    'answer': 'hoyed',
    'preTestVideos': ['hVds/F1/hoyed_1.mp4', 'hVds/M2/hoyed_1.mp4'],
    'trainVideos': ['hVds/F1/hoyed_2.mp4', 'hVds/M2/hoyed_2.mp4'],
    'testVideos': ['hVds/F1/hoyed_3.mp4', 'hVds/M2/hoyed_3.mp4'],
  },
  'hide': {
    'answer': 'hide',
    'preTestVideos': ['hVds/F1/hide_1.mp4', 'hVds/M2/hide_1.mp4'],
    'trainVideos': ['hVds/F1/hide_2.mp4', 'hVds/M2/hide_2.mp4'],
    'testVideos': ['hVds/F1/hide_3.mp4', 'hVds/M2/hide_3.mp4'],
  },
  'who\'d': {
    'answer': 'who\'d',
    'preTestVideos': ['hVds/F1/who\'d_1.mp4', 'hVds/M2/who\'d_1.mp4'],
    'trainVideos': ['hVds/F1/who\'d_2.mp4', 'hVds/M2/who\'d_2.mp4'],
    'testVideos': ['hVds/F1/who\'d_3.mp4', 'hVds/M2/who\'d_3.mp4'],
  },
  'heed': {
    'answer': 'heed',
    'preTestVideos': ['hVds/F1/heed_1.mp4', 'hVds/M2/heed_1.mp4'],
    'trainVideos': ['hVds/F1/heed_2.mp4', 'hVds/M2/heed_2.mp4'],
    'testVideos': ['hVds/F1/heed_3.mp4', 'hVds/M2/heed_3.mp4'],
  },
  'hood': {
    'answer': 'hood',
    'preTestVideos': ['hVds/F1/hood_1.mp4', 'hVds/M2/hood_1.mp4'],
    'trainVideos': ['hVds/F1/hood_2.mp4', 'hVds/M2/hood_2.mp4'],
    'testVideos': ['hVds/F1/hood_3.mp4', 'hVds/M2/hood_3.mp4'],
  },
  'hid': {
    'answer': 'hid',
    'preTestVideos': ['hVds/F1/hid_1.mp4', 'hVds/M2/hid_1.mp4'],
    'trainVideos': ['hVds/F1/hid_2.mp4', 'hVds/M2/hid_2.mp4'],
    'testVideos': ['hVds/F1/hid_3.mp4', 'hVds/M2/hid_3.mp4'],
  },
  'hayed': {
    'answer': 'hayed',
    'preTestVideos': ['hVds/F1/hayed_1.mp4', 'hVds/M2/hayed_1.mp4'],
    'trainVideos': ['hVds/F1/hayed_2.mp4', 'hVds/M2/hayed_2.mp4'],
    'testVideos': ['hVds/F1/hayed_3.mp4', 'hVds/M2/hayed_3.mp4'],
  },
  'how\'d': {
    'answer': 'how\'d',
    'preTestVideos': ['hVds/F1/how\'d_1.mp4', 'hVds/M2/how\'d_1.mp4'],
    'trainVideos': ['hVds/F1/how\'d_2.mp4', 'hVds/M2/how\'d_2.mp4'],
    'testVideos': ['hVds/F1/how\'d_3.mp4', 'hVds/M2/how\'d_3.mp4'],
  },
  'hod': {
    'answer': 'hod',
    'preTestVideos': ['hVds/F1/hod_1.mp4', 'hVds/M2/hod_1.mp4'],
    'trainVideos': ['hVds/F1/hod_2.mp4', 'hVds/M2/hod_2.mp4'],
    'testVideos': ['hVds/F1/hod_3.mp4', 'hVds/M2/hod_3.mp4'],
  },
  'hud': {
    'answer': 'hud',
    'preTestVideos': ['hVds/F1/hud_1.mp4', 'hVds/M2/hud_1.mp4'],
    'trainVideos': ['hVds/F1/hud_2.mp4', 'hVds/M2/hud_2.mp4'],
    'testVideos': ['hVds/F1/hud_3.mp4', 'hVds/M2/hud_3.mp4'],
  },
  'had': {
    'answer': 'had',
    'preTestVideos': ['hVds/F1/had_1.mp4', 'hVds/M2/had_1.mp4'],
    'trainVideos': ['hVds/F1/had_2.mp4', 'hVds/M2/had_2.mp4'],
    'testVideos': ['hVds/F1/had_3.mp4', 'hVds/M2/had_3.mp4'],
  },
  'sentence_1': {
    'answer': 'Peter ordered eight heavy desks',
    'testVideos':
        ['Structured Sentences/F1/Peter ordered eight heavy desks_1.mp4'],
  },
  'sentence_2': {
    'answer': 'Peter prefers nineteen dark tables',
    'testVideos':
        ['Structured Sentences/F1/Peter prefers nineteen dark tables_1.mp4'],
  },
  'sentence_3': {
    'answer': 'Steven got seven green chairs',
    'testVideos':
        ['Structured Sentences/F1/Steven got seven green chairs_1.mp4'],
  },
  'sentence_4': {
    'answer': 'Peter has nineteen large desks',
    'testVideos':
        ['Structured Sentences/F1/Peter has nineteen large desks_1.mp4'],
  },
  'sentence_5': {
    'answer': 'Thomas prefers three white desks',
    'testVideos':
        ['Structured Sentences/F1/Thomas prefers three white desks_1.mp4'],
  },
  'sentence_6': {
    'answer': 'Peter got twelve old rings',
    'testVideos': ['Structured Sentences/F1/Peter got twelve old rings_1.mp4'],
  },
  'sentence_7': {
    'answer': 'Peter sold four large flowers',
    'testVideos':
        ['Structured Sentences/F1/Peter sold four large flowers_1.mp4'],
  },
  'sentence_8': {
    'answer': 'Nina has sixty green desks',
    'testVideos': ['Structured Sentences/F1/Nina has sixty green desks_1.mp4'],
  },
  'sentence_9': {
    'answer': 'Nina ordered sixty heavy sofas',
    'testVideos':
        ['Structured Sentences/F1/Nina ordered sixty heavy sofas_1.mp4'],
  },
  'sentence_10': {
    'answer': 'William kept nine small windows',
    'testVideos':
        ['Structured Sentences/F1/William kept nine small windows_1.mp4'],
  },
  'sentence_11': {
    'answer': 'Doris has four old sofas',
    'testVideos': ['Structured Sentences/F1/Doris has four old sofas_1.mp4'],
  },
  'sentence_12': {
    'answer': 'Nina prefers four red windows',
    'testVideos':
        ['Structured Sentences/F1/Nina prefers four red windows_1.mp4'],
  },
  'sentence_13': {
    'answer': 'Peter prefers two heavy desks',
    'testVideos':
        ['Structured Sentences/F1/Peter prefers two heavy desks_1.mp4'],
  },
  'sentence_14': {
    'answer': 'Kathy ordered eight old sofas',
    'testVideos':
        ['Structured Sentences/F1/Kathy ordered eight old sofas_1.mp4'],
  },
  'sentence_15': {
    'answer': 'Lucy got seven large rings',
    'testVideos': ['Structured Sentences/F1/Lucy got seven large rings_1.mp4'],
  },
  'sentence_16': {
    'answer': 'Rachel ordered four pretty spoons',
    'testVideos':
        ['Structured Sentences/F1/Rachel ordered four pretty spoons_1.mp4'],
  },
  'sentence_17': {
    'answer': 'Peter gives eight large desks',
    'testVideos':
        ['Structured Sentences/F1/Peter gives eight large desks_1.mp4'],
  },
  'sentence_18': {
    'answer': 'Rachel ordered twelve cheap tables',
    'testVideos':
        ['Structured Sentences/F1/Rachel ordered twelve cheap tables_1.mp4'],
  },
  'sentence_19': {
    'answer': 'William sold three green desks',
    'testVideos':
        ['Structured Sentences/F1/William sold three green desks_1.mp4'],
  },
  'sentence_20': {
    'answer': 'Lucy got three red desks',
    'testVideos': ['Structured Sentences/F1/Lucy got three red desks_1.mp4'],
  },
  'sentence_21': {
    'answer': 'Nina gives sixty small desks',
    'testVideos':
        ['Structured Sentences/F1/Nina gives sixty small desks_1.mp4'],
  },
  'sentence_22': {
    'answer': 'William got three dark houses',
    'testVideos':
        ['Structured Sentences/F1/William got three dark houses_1.mp4'],
  },
  'sentence_23': {
    'answer': 'Steven bought nine heavy toys',
    'testVideos':
        ['Structured Sentences/F1/Steven bought nine heavy toys_1.mp4'],
  },
  'sentence_24': {
    'answer': 'Allen sees four green houses',
    'testVideos':
        ['Structured Sentences/F1/Allen sees four green houses_1.mp4'],
  },
  'sentence_25': {
    'answer': 'Steven gives nine red desks',
    'testVideos': ['Structured Sentences/F1/Steven gives nine red desks_1.mp4'],
  },
  'sentence_26': {
    'answer': 'Thomas got sixty cheap desks',
    'testVideos':
        ['Structured Sentences/F1/Thomas got sixty cheap desks_1.mp4'],
  },
  'sentence_27': {
    'answer': 'Doris kept nineteen cheap spoons',
    'testVideos':
        ['Structured Sentences/F1/Doris kept nineteen cheap spoons_1.mp4'],
  },
  'sentence_28': {
    'answer': 'Peter got three green desks',
    'testVideos': ['Structured Sentences/F1/Peter got three green desks_1.mp4'],
  },
  'sentence_29': {
    'answer': 'Allen sees fifteen red tables',
    'testVideos':
        ['Structured Sentences/F1/Allen sees fifteen red tables_1.mp4'],
  },
  'sentence_30': {
    'answer': 'Kathy wants two pretty sofas',
    'testVideos':
        ['Structured Sentences/F1/Kathy wants two pretty sofas_1.mp4'],
  },
  'sentence_31': {
    'answer': 'Allen sees four dark toys',
    'testVideos': ['Structured Sentences/F1/Allen sees four dark toys_1.mp4'],
  },
  'sentence_32': {
    'answer': 'Rachel got three small sofas',
    'testVideos':
        ['Structured Sentences/F1/Rachel got three small sofas_1.mp4'],
  },
  'sentence_33': {
    'answer': 'Kathy sold nineteen old sofas',
    'testVideos':
        ['Structured Sentences/F1/Kathy sold nineteen old sofas_1.mp4'],
  },
  'sentence_34': {
    'answer': 'Allen wants two white flowers',
    'testVideos':
        ['Structured Sentences/F1/Allen wants two white flowers_1.mp4'],
  },
  'sentence_35': {
    'answer': 'Peter gives fifteen cheap windows',
    'testVideos':
        ['Structured Sentences/F1/Peter gives fifteen cheap windows_1.mp4'],
  },
  'sentence_36': {
    'answer': 'Peter gives eight red houses',
    'testVideos':
        ['Structured Sentences/F1/Peter gives eight red houses_1.mp4'],
  },
  'sentence_37': {
    'answer': 'Peter got fifteen heavy tables',
    'testVideos':
        ['Structured Sentences/F1/Peter got fifteen heavy tables_1.mp4'],
  },
  'sentence_38': {
    'answer': 'Thomas got nineteen heavy desks',
    'testVideos':
        ['Structured Sentences/F1/Thomas got nineteen heavy desks_1.mp4'],
  },
  'sentence_39': {
    'answer': 'Allen prefers two old desks',
    'testVideos': ['Structured Sentences/F1/Allen prefers two old desks_1.mp4'],
  },
  'sentence_40': {
    'answer': 'Peter has fifteen green rings',
    'testVideos':
        ['Structured Sentences/F1/Peter has fifteen green rings_1.mp4'],
  },
  'sentence_41': {
    'answer': 'Peter gives four green sofas',
    'testVideos':
        ['Structured Sentences/F1/Peter gives four green sofas_1.mp4'],
  },
  'sentence_42': {
    'answer': 'William ordered sixty cheap toys',
    'testVideos':
        ['Structured Sentences/F1/William ordered sixty cheap toys_1.mp4'],
  },
  'sentence_43': {
    'answer': 'Rachel sees three large spoons',
    'testVideos':
        ['Structured Sentences/F1/Rachel sees three large spoons_1.mp4'],
  },
  'sentence_44': {
    'answer': 'Thomas ordered fifteen red houses',
    'testVideos':
        ['Structured Sentences/F1/Thomas ordered fifteen red houses_1.mp4'],
  },
  'sentence_45': {
    'answer': 'Peter wants seven white spoons',
    'testVideos':
        ['Structured Sentences/F1/Peter wants seven white spoons_1.mp4'],
  },
  'sentence_46': {
    'answer': 'Peter kept three large chairs',
    'testVideos':
        ['Structured Sentences/F1/Peter kept three large chairs_1.mp4'],
  },
  'sentence_47': {
    'answer': 'Kathy prefers seven green rings',
    'testVideos':
        ['Structured Sentences/F1/Kathy prefers seven green rings_1.mp4'],
  },
  'sentence_48': {
    'answer': 'Peter ordered twelve red windows',
    'testVideos':
        ['Structured Sentences/F1/Peter ordered twelve red windows_1.mp4'],
  },
  'sentence_49': {
    'answer': 'Thomas sees fifteen cheap windows',
    'testVideos':
        ['Structured Sentences/F1/Thomas sees fifteen cheap windows_1.mp4'],
  },
  'sentence_50': {
    'answer': 'Steven got seven green flowers',
    'testVideos':
        ['Structured Sentences/F1/Steven got seven green flowers_1.mp4'],
  },
  'sentence_51': {
    'answer': 'Lucy wants eight cheap tables',
    'testVideos':
        ['Structured Sentences/F1/Lucy wants eight cheap tables_1.mp4'],
  },
  'sentence_52': {
    'answer': 'Allen sold seven old spoons',
    'testVideos': ['Structured Sentences/F1/Allen sold seven old spoons_1.mp4'],
  },
  'sentence_53': {
    'answer': 'Rachel has nineteen white desks',
    'testVideos':
        ['Structured Sentences/F1/Rachel has nineteen white desks_1.mp4'],
  },
  'sentence_54': {
    'answer': 'Peter ordered two dark tables',
    'testVideos':
        ['Structured Sentences/F1/Peter ordered two dark tables_1.mp4'],
  },
  'sentence_55': {
    'answer': 'Thomas prefers nineteen white spoons',
    'testVideos':
        ['Structured Sentences/F1/Thomas prefers nineteen white spoons_1.mp4'],
  },
  'sentence_56': {
    'answer': 'Allen sold twelve dark desks',
    'testVideos':
        ['Structured Sentences/F1/Allen sold twelve dark desks_1.mp4'],
  },
  'sentence_57': {
    'answer': 'Steven got fifteen old chairs',
    'testVideos':
        ['Structured Sentences/F1/Steven got fifteen old chairs_1.mp4'],
  },
  'sentence_58': {
    'answer': 'William gives nineteen dark rings',
    'testVideos':
        ['Structured Sentences/F1/William gives nineteen dark rings_1.mp4'],
  },
  'sentence_59': {
    'answer': 'Lucy ordered three large tables',
    'testVideos':
        ['Structured Sentences/F1/Lucy ordered three large tables_1.mp4'],
  },
  'sentence_60': {
    'answer': 'Peter has three large desks',
    'testVideos': ['Structured Sentences/F1/Peter has three large desks_1.mp4'],
  },
  'sentence_61': {
    'answer': 'Peter has two cheap windows',
    'testVideos': ['Structured Sentences/F1/Peter has two cheap windows_1.mp4'],
  },
  'sentence_62': {
    'answer': 'Doris sold sixty green spoons',
    'testVideos':
        ['Structured Sentences/F1/Doris sold sixty green spoons_1.mp4'],
  },
  'sentence_63': {
    'answer': 'Lucy gives nineteen pretty desks',
    'testVideos':
        ['Structured Sentences/F1/Lucy gives nineteen pretty desks_1.mp4'],
  },
  'sentence_64': {
    'answer': 'Nina ordered three heavy windows',
    'testVideos':
        ['Structured Sentences/F1/Nina ordered three heavy windows_1.mp4'],
  },
  'sentence_65': {
    'answer': 'Thomas bought three large rings',
    'testVideos':
        ['Structured Sentences/F1/Thomas bought three large rings_1.mp4'],
  },
  'sentence_66': {
    'answer': 'Thomas kept two small spoons',
    'testVideos':
        ['Structured Sentences/F1/Thomas kept two small spoons_1.mp4'],
  },
  'sentence_67': {
    'answer': 'Kathy sold twelve green chairs',
    'testVideos':
        ['Structured Sentences/F1/Kathy sold twelve green chairs_1.mp4'],
  },
  'sentence_68': {
    'answer': 'Peter gives sixty pretty rings',
    'testVideos':
        ['Structured Sentences/F1/Peter gives sixty pretty rings_1.mp4'],
  },
  'sentence_69': {
    'answer': 'Thomas prefers nine cheap spoons',
    'testVideos':
        ['Structured Sentences/F1/Thomas prefers nine cheap spoons_1.mp4'],
  },
  'sentence_70': {
    'answer': 'Peter wants fifteen dark desks',
    'testVideos':
        ['Structured Sentences/F1/Peter wants fifteen dark desks_1.mp4'],
  },
  'sentence_71': {
    'answer': 'Steven got three small tables',
    'testVideos':
        ['Structured Sentences/F1/Steven got three small tables_1.mp4'],
  },
  'sentence_72': {
    'answer': 'Steven gives three dark chairs',
    'testVideos':
        ['Structured Sentences/F1/Steven gives three dark chairs_1.mp4'],
  },
  'sentence_73': {
    'answer': 'Thomas sold nine large desks',
    'testVideos':
        ['Structured Sentences/F1/Thomas sold nine large desks_1.mp4'],
  },
  'sentence_74': {
    'answer': 'Doris got seven old windows',
    'testVideos': ['Structured Sentences/F1/Doris got seven old windows_1.mp4'],
  },
  'sentence_75': {
    'answer': 'Allen kept nine large tables',
    'testVideos':
        ['Structured Sentences/F1/Allen kept nine large tables_1.mp4'],
  },
  'sentence_76': {
    'answer': 'Doris sold nine cheap houses',
    'testVideos':
        ['Structured Sentences/F1/Doris sold nine cheap houses_1.mp4'],
  },
  'sentence_77': {
    'answer': 'Rachel has twelve old tables',
    'testVideos':
        ['Structured Sentences/F1/Rachel has twelve old tables_1.mp4'],
  },
  'sentence_78': {
    'answer': 'Steven sees eight green sofas',
    'testVideos':
        ['Structured Sentences/F1/Steven sees eight green sofas_1.mp4'],
  },
  'sentence_79': {
    'answer': 'Peter gives fifteen large tables',
    'testVideos':
        ['Structured Sentences/F1/Peter gives fifteen large tables_1.mp4'],
  },
  'sentence_80': {
    'answer': 'Doris ordered twelve old flowers',
    'testVideos':
        ['Structured Sentences/F1/Doris ordered twelve old flowers_1.mp4'],
  },
  'sentence_81': {
    'answer': 'Peter got seven white windows',
    'testVideos':
        ['Structured Sentences/F1/Peter got seven white windows_1.mp4'],
  },
  'sentence_82': {
    'answer': 'Peter got nine dark spoons',
    'testVideos': ['Structured Sentences/F1/Peter got nine dark spoons_1.mp4'],
  },
  'sentence_83': {
    'answer': 'Rachel got three large chairs',
    'testVideos':
        ['Structured Sentences/F1/Rachel got three large chairs_1.mp4'],
  },
  'sentence_84': {
    'answer': 'Kathy kept three cheap flowers',
    'testVideos':
        ['Structured Sentences/F1/Kathy kept three cheap flowers_1.mp4'],
  },
  'sentence_85': {
    'answer': 'William kept three small spoons',
    'testVideos':
        ['Structured Sentences/F1/William kept three small spoons_1.mp4'],
  },
  'sentence_86': {
    'answer': 'Allen sees three small rings',
    'testVideos':
        ['Structured Sentences/F1/Allen sees three small rings_1.mp4'],
  },
  'sentence_87': {
    'answer': 'Peter has nineteen red spoons',
    'testVideos':
        ['Structured Sentences/F1/Peter has nineteen red spoons_1.mp4'],
  },
  'sentence_88': {
    'answer': 'Rachel sold three red houses',
    'testVideos':
        ['Structured Sentences/F1/Rachel sold three red houses_1.mp4'],
  },
  'sentence_89': {
    'answer': 'Allen has nineteen large spoons',
    'testVideos':
        ['Structured Sentences/F1/Allen has nineteen large spoons_1.mp4'],
  },
  'sentence_90': {
    'answer': 'Rachel bought two large desks',
    'testVideos':
        ['Structured Sentences/F1/Rachel bought two large desks_1.mp4'],
  },
  'sentence_91': {
    'answer': 'Peter bought nineteen cheap houses',
    'testVideos':
        ['Structured Sentences/F1/Peter bought nineteen cheap houses_1.mp4'],
  },
  'sentence_92': {
    'answer': 'Doris wants eight old tables',
    'testVideos':
        ['Structured Sentences/F1/Doris wants eight old tables_1.mp4'],
  },
  'sentence_93': {
    'answer': 'Steven got eight dark chairs',
    'testVideos':
        ['Structured Sentences/F1/Steven got eight dark chairs_1.mp4'],
  },
  'sentence_94': {
    'answer': 'William has sixty white chairs',
    'testVideos':
        ['Structured Sentences/F1/William has sixty white chairs_1.mp4'],
  },
  'sentence_95': {
    'answer': 'Doris sees sixty pretty desks',
    'testVideos':
        ['Structured Sentences/F1/Doris sees sixty pretty desks_1.mp4'],
  },
  'sentence_96': {
    'answer': 'Thomas prefers nine white tables',
    'testVideos':
        ['Structured Sentences/F1/Thomas prefers nine white tables_1.mp4'],
  },
  'sentence_97': {
    'answer': 'Steven sold nine green sofas',
    'testVideos':
        ['Structured Sentences/F1/Steven sold nine green sofas_1.mp4'],
  },
  'sentence_98': {
    'answer': 'Thomas ordered four dark houses',
    'testVideos':
        ['Structured Sentences/F1/Thomas ordered four dark houses_1.mp4'],
  },
  'sentence_99': {
    'answer': 'Steven ordered two dark spoons',
    'testVideos':
        ['Structured Sentences/F1/Steven ordered two dark spoons_1.mp4'],
  },
  'sentence_100': {
    'answer': 'Peter got four green desks',
    'testVideos': ['Structured Sentences/F1/Peter got four green desks_1.mp4'],
  },
  'sentence_101': {
    'answer': 'Doris prefers three red tables',
    'testVideos':
        ['Structured Sentences/F1/Doris prefers three red tables_1.mp4'],
  },
  'sentence_102': {
    'answer': 'Steven ordered seven small chairs',
    'testVideos':
        ['Structured Sentences/F1/Steven ordered seven small chairs_1.mp4'],
  },
  'sentence_103': {
    'answer': 'Peter wants two white chairs',
    'testVideos':
        ['Structured Sentences/F1/Peter wants two white chairs_1.mp4'],
  },
  'sentence_104': {
    'answer': 'Thomas prefers eight cheap toys',
    'testVideos':
        ['Structured Sentences/F1/Thomas prefers eight cheap toys_1.mp4'],
  },
  'sentence_105': {
    'answer': 'Peter ordered three red houses',
    'testVideos':
        ['Structured Sentences/F1/Peter ordered three red houses_1.mp4'],
  },
  'sentence_106': {
    'answer': 'Kathy got three old flowers',
    'testVideos': ['Structured Sentences/F1/Kathy got three old flowers_1.mp4'],
  },
  'sentence_107': {
    'answer': 'Allen prefers four small sofas',
    'testVideos':
        ['Structured Sentences/F1/Allen prefers four small sofas_1.mp4'],
  },
  'sentence_108': {
    'answer': 'Peter sold fifteen cheap spoons',
    'testVideos':
        ['Structured Sentences/F1/Peter sold fifteen cheap spoons_1.mp4'],
  },
  'sentence_109': {
    'answer': 'Peter sold nineteen pretty rings',
    'testVideos':
        ['Structured Sentences/F1/Peter sold nineteen pretty rings_1.mp4'],
  },
  'sentence_110': {
    'answer': 'Doris ordered nineteen white toys',
    'testVideos':
        ['Structured Sentences/F1/Doris ordered nineteen white toys_1.mp4'],
  },
  'sentence_111': {
    'answer': 'Rachel prefers three pretty spoons',
    'testVideos':
        ['Structured Sentences/F1/Rachel prefers three pretty spoons_1.mp4'],
  },
  'sentence_112': {
    'answer': 'Lucy prefers four large flowers',
    'testVideos':
        ['Structured Sentences/F1/Lucy prefers four large flowers_1.mp4'],
  },
  'sentence_113': {
    'answer': 'Peter got twelve large windows',
    'testVideos':
        ['Structured Sentences/F1/Peter got twelve large windows_1.mp4'],
  },
  'sentence_114': {
    'answer': 'Lucy has two white desks',
    'testVideos': ['Structured Sentences/F1/Lucy has two white desks_1.mp4'],
  },
  'sentence_115': {
    'answer': 'Kathy wants eight dark flowers',
    'testVideos':
        ['Structured Sentences/F1/Kathy wants eight dark flowers_1.mp4'],
  },
  'sentence_116': {
    'answer': 'Kathy ordered fifteen cheap toys',
    'testVideos':
        ['Structured Sentences/F1/Kathy ordered fifteen cheap toys_1.mp4'],
  },
  'sentence_117': {
    'answer': 'Steven kept three green chairs',
    'testVideos':
        ['Structured Sentences/F1/Steven kept three green chairs_1.mp4'],
  },
  'sentence_118': {
    'answer': 'Kathy sees four old toys',
    'testVideos': ['Structured Sentences/F1/Kathy sees four old toys_1.mp4'],
  },
  'sentence_119': {
    'answer': 'Nina kept twelve pretty tables',
    'testVideos':
        ['Structured Sentences/F1/Nina kept twelve pretty tables_1.mp4'],
  },
  'sentence_120': {
    'answer': 'Rachel prefers twelve red desks',
    'testVideos':
        ['Structured Sentences/F1/Rachel prefers twelve red desks_1.mp4'],
  },
  'sentence_121': {
    'answer': 'Peter ordered four white desks',
    'testVideos':
        ['Structured Sentences/F1/Peter ordered four white desks_1.mp4'],
  },
  'sentence_122': {
    'answer': 'Peter got fifteen old tables',
    'testVideos':
        ['Structured Sentences/F1/Peter got fifteen old tables_1.mp4'],
  },
  'sentence_123': {
    'answer': 'Lucy got four pretty sofas',
    'testVideos': ['Structured Sentences/F1/Lucy got four pretty sofas_1.mp4'],
  },
  'sentence_124': {
    'answer': 'Nina kept four pretty houses',
    'testVideos':
        ['Structured Sentences/F1/Nina kept four pretty houses_1.mp4'],
  },
  'sentence_125': {
    'answer': 'Thomas has twelve heavy sofas',
    'testVideos':
        ['Structured Sentences/F1/Thomas has twelve heavy sofas_1.mp4'],
  },
  'sentence_126': {
    'answer': 'Allen gives fifteen cheap flowers',
    'testVideos':
        ['Structured Sentences/F1/Allen gives fifteen cheap flowers_1.mp4'],
  },
  'sentence_127': {
    'answer': 'Thomas wants four pretty rings',
    'testVideos':
        ['Structured Sentences/F1/Thomas wants four pretty rings_1.mp4'],
  },
  'sentence_128': {
    'answer': 'Doris sees fifteen large sofas',
    'testVideos':
        ['Structured Sentences/F1/Doris sees fifteen large sofas_1.mp4'],
  },
  'sentence_129': {
    'answer': 'Peter has fifteen white windows',
    'testVideos':
        ['Structured Sentences/F1/Peter has fifteen white windows_1.mp4'],
  },
  'sentence_130': {
    'answer': 'Nina has two heavy tables',
    'testVideos': ['Structured Sentences/F1/Nina has two heavy tables_1.mp4'],
  },
  'sentence_131': {
    'answer': 'Steven sold twelve large desks',
    'testVideos':
        ['Structured Sentences/F1/Steven sold twelve large desks_1.mp4'],
  },
  'sentence_132': {
    'answer': 'Peter has four large toys',
    'testVideos': ['Structured Sentences/F1/Peter has four large toys_1.mp4'],
  },
  'sentence_133': {
    'answer': 'Peter ordered eight white toys',
    'testVideos':
        ['Structured Sentences/F1/Peter ordered eight white toys_1.mp4'],
  },
  'sentence_134': {
    'answer': 'Rachel kept sixty large desks',
    'testVideos':
        ['Structured Sentences/F1/Rachel kept sixty large desks_1.mp4'],
  },
  'sentence_135': {
    'answer': 'Peter wants nine large toys',
    'testVideos': ['Structured Sentences/F1/Peter wants nine large toys_1.mp4'],
  },
  'sentence_136': {
    'answer': 'Peter got twelve large toys',
    'testVideos': ['Structured Sentences/F1/Peter got twelve large toys_1.mp4'],
  },
  'sentence_137': {
    'answer': 'Peter kept sixty pretty toys',
    'testVideos':
        ['Structured Sentences/F1/Peter kept sixty pretty toys_1.mp4'],
  },
  'sentence_138': {
    'answer': 'William has nine small flowers',
    'testVideos':
        ['Structured Sentences/F1/William has nine small flowers_1.mp4'],
  },
  'sentence_139': {
    'answer': 'Kathy gives nine old chairs',
    'testVideos': ['Structured Sentences/F1/Kathy gives nine old chairs_1.mp4'],
  },
  'sentence_140': {
    'answer': 'Allen sold sixty pretty chairs',
    'testVideos':
        ['Structured Sentences/F1/Allen sold sixty pretty chairs_1.mp4'],
  },
  'sentence_141': {
    'answer': 'William prefers sixty small sofas',
    'testVideos':
        ['Structured Sentences/F1/William prefers sixty small sofas_1.mp4'],
  },
  'sentence_142': {
    'answer': 'Rachel gives fifteen small desks',
    'testVideos':
        ['Structured Sentences/F1/Rachel gives fifteen small desks_1.mp4'],
  },
  'sentence_143': {
    'answer': 'Steven ordered three large rings',
    'testVideos':
        ['Structured Sentences/F1/Steven ordered three large rings_1.mp4'],
  },
  'sentence_144': {
    'answer': 'Nina ordered nine heavy tables',
    'testVideos':
        ['Structured Sentences/F1/Nina ordered nine heavy tables_1.mp4'],
  },
  'sentence_145': {
    'answer': 'William prefers three small houses',
    'testVideos':
        ['Structured Sentences/F1/William prefers three small houses_1.mp4'],
  },
  'sentence_146': {
    'answer': 'William got sixty dark chairs',
    'testVideos':
        ['Structured Sentences/F1/William got sixty dark chairs_1.mp4'],
  },
  'sentence_147': {
    'answer': 'William ordered three old desks',
    'testVideos':
        ['Structured Sentences/F1/William ordered three old desks_1.mp4'],
  },
  'sentence_148': {
    'answer': 'Kathy wants sixty old toys',
    'testVideos': ['Structured Sentences/F1/Kathy wants sixty old toys_1.mp4'],
  },
  'sentence_149': {
    'answer': 'Doris got fifteen cheap desks',
    'testVideos':
        ['Structured Sentences/F1/Doris got fifteen cheap desks_1.mp4'],
  },
  'sentence_150': {
    'answer': 'William prefers two small desks',
    'testVideos':
        ['Structured Sentences/F1/William prefers two small desks_1.mp4'],
  },
  'sentence_151': {
    'answer': 'Nina got two heavy tables',
    'testVideos': ['Structured Sentences/F1/Nina got two heavy tables_1.mp4'],
  },
  'sentence_152': {
    'answer': 'Steven sees four dark spoons',
    'testVideos':
        ['Structured Sentences/F1/Steven sees four dark spoons_1.mp4'],
  },
  'sentence_153': {
    'answer': 'Lucy has nineteen dark desks',
    'testVideos':
        ['Structured Sentences/F1/Lucy has nineteen dark desks_1.mp4'],
  },
  'sentence_154': {
    'answer': 'Peter sees three red flowers',
    'testVideos':
        ['Structured Sentences/F1/Peter sees three red flowers_1.mp4'],
  },
  'sentence_155': {
    'answer': 'Doris ordered two cheap sofas',
    'testVideos':
        ['Structured Sentences/F1/Doris ordered two cheap sofas_1.mp4'],
  },
  'sentence_156': {
    'answer': 'Allen got sixty cheap toys',
    'testVideos': ['Structured Sentences/F1/Allen got sixty cheap toys_1.mp4'],
  },
  'sentence_157': {
    'answer': 'Kathy ordered three cheap spoons',
    'testVideos':
        ['Structured Sentences/F1/Kathy ordered three cheap spoons_1.mp4'],
  },
  'sentence_158': {
    'answer': 'Kathy bought fifteen large toys',
    'testVideos':
        ['Structured Sentences/F1/Kathy bought fifteen large toys_1.mp4'],
  },
  'sentence_159': {
    'answer': 'Nina sold sixty large rings',
    'testVideos': ['Structured Sentences/F1/Nina sold sixty large rings_1.mp4'],
  },
  'sentence_160': {
    'answer': 'Lucy sold nine old chairs',
    'testVideos': ['Structured Sentences/F1/Lucy sold nine old chairs_1.mp4'],
  },
  'sentence_161': {
    'answer': 'Allen sees nine large chairs',
    'testVideos':
        ['Structured Sentences/F1/Allen sees nine large chairs_1.mp4'],
  },
  'sentence_162': {
    'answer': 'Peter sees sixty large spoons',
    'testVideos':
        ['Structured Sentences/F1/Peter sees sixty large spoons_1.mp4'],
  },
  'sentence_163': {
    'answer': 'William sold two heavy rings',
    'testVideos':
        ['Structured Sentences/F1/William sold two heavy rings_1.mp4'],
  },
  'sentence_164': {
    'answer': 'Doris bought eight cheap houses',
    'testVideos':
        ['Structured Sentences/F1/Doris bought eight cheap houses_1.mp4'],
  },
  'sentence_165': {
    'answer': 'Peter sold sixty white desks',
    'testVideos':
        ['Structured Sentences/F1/Peter sold sixty white desks_1.mp4'],
  },
  'sentence_166': {
    'answer': 'Lucy wants twelve white tables',
    'testVideos':
        ['Structured Sentences/F1/Lucy wants twelve white tables_1.mp4'],
  },
  'sentence_167': {
    'answer': 'William ordered twelve cheap rings',
    'testVideos':
        ['Structured Sentences/F1/William ordered twelve cheap rings_1.mp4'],
  },
  'sentence_168': {
    'answer': 'Doris has seven green tables',
    'testVideos':
        ['Structured Sentences/F1/Doris has seven green tables_1.mp4'],
  },
  'sentence_169': {
    'answer': 'Peter sold twelve small flowers',
    'testVideos':
        ['Structured Sentences/F1/Peter sold twelve small flowers_1.mp4'],
  },
  'sentence_170': {
    'answer': 'William sold sixty large desks',
    'testVideos':
        ['Structured Sentences/F1/William sold sixty large desks_1.mp4'],
  },
  'sentence_171': {
    'answer': 'Nina sold two heavy desks',
    'testVideos': ['Structured Sentences/F1/Nina sold two heavy desks_1.mp4'],
  },
  'sentence_172': {
    'answer': 'Allen has sixty large windows',
    'testVideos':
        ['Structured Sentences/F1/Allen has sixty large windows_1.mp4'],
  },
  'sentence_173': {
    'answer': 'Peter got seven white tables',
    'testVideos':
        ['Structured Sentences/F1/Peter got seven white tables_1.mp4'],
  },
  'sentence_174': {
    'answer': 'Rachel sees nine pretty desks',
    'testVideos':
        ['Structured Sentences/F1/Rachel sees nine pretty desks_1.mp4'],
  },
  'sentence_175': {
    'answer': 'Peter got nine green sofas',
    'testVideos': ['Structured Sentences/F1/Peter got nine green sofas_1.mp4'],
  },
  'sentence_176': {
    'answer': 'Peter sold two large flowers',
    'testVideos':
        ['Structured Sentences/F1/Peter sold two large flowers_1.mp4'],
  },
  'sentence_177': {
    'answer': 'Thomas bought sixty white spoons',
    'testVideos':
        ['Structured Sentences/F1/Thomas bought sixty white spoons_1.mp4'],
  },
  'sentence_178': {
    'answer': 'Thomas sees eight green rings',
    'testVideos':
        ['Structured Sentences/F1/Thomas sees eight green rings_1.mp4'],
  },
  'sentence_179': {
    'answer': 'Steven got seven green desks',
    'testVideos':
        ['Structured Sentences/F1/Steven got seven green desks_1.mp4'],
  },
  'sentence_180': {
    'answer': 'Kathy bought three red toys',
    'testVideos': ['Structured Sentences/F1/Kathy bought three red toys_1.mp4'],
  },
  'sentence_181': {
    'answer': 'Kathy bought sixty red chairs',
    'testVideos':
        ['Structured Sentences/F1/Kathy bought sixty red chairs_1.mp4'],
  },
  'sentence_182': {
    'answer': 'Doris sees twelve large desks',
    'testVideos':
        ['Structured Sentences/F1/Doris sees twelve large desks_1.mp4'],
  },
  'sentence_183': {
    'answer': 'Allen sees twelve large chairs',
    'testVideos':
        ['Structured Sentences/F1/Allen sees twelve large chairs_1.mp4'],
  },
  'sentence_184': {
    'answer': 'William gives three cheap windows',
    'testVideos':
        ['Structured Sentences/F1/William gives three cheap windows_1.mp4'],
  },
  'sentence_185': {
    'answer': 'Rachel got sixty pretty flowers',
    'testVideos':
        ['Structured Sentences/F1/Rachel got sixty pretty flowers_1.mp4'],
  },
  'sentence_186': {
    'answer': 'Kathy gives seven old desks',
    'testVideos': ['Structured Sentences/F1/Kathy gives seven old desks_1.mp4'],
  },
  'sentence_187': {
    'answer': 'Kathy gives twelve cheap tables',
    'testVideos':
        ['Structured Sentences/F1/Kathy gives twelve cheap tables_1.mp4'],
  },
  'sentence_188': {
    'answer': 'William kept three green desks',
    'testVideos':
        ['Structured Sentences/F1/William kept three green desks_1.mp4'],
  },
  'sentence_189': {
    'answer': 'Rachel kept nine dark desks',
    'testVideos': ['Structured Sentences/F1/Rachel kept nine dark desks_1.mp4'],
  },
  'sentence_190': {
    'answer': 'Thomas gives three small houses',
    'testVideos':
        ['Structured Sentences/F1/Thomas gives three small houses_1.mp4'],
  },
  'sentence_191': {
    'answer': 'Steven ordered twelve dark sofas',
    'testVideos':
        ['Structured Sentences/F1/Steven ordered twelve dark sofas_1.mp4'],
  },
  'sentence_192': {
    'answer': 'Thomas wants three old flowers',
    'testVideos':
        ['Structured Sentences/F1/Thomas wants three old flowers_1.mp4'],
  },
  'sentence_193': {
    'answer': 'Kathy got three small desks',
    'testVideos': ['Structured Sentences/F1/Kathy got three small desks_1.mp4'],
  },
  'sentence_194': {
    'answer': 'Steven kept nine small spoons',
    'testVideos':
        ['Structured Sentences/F1/Steven kept nine small spoons_1.mp4'],
  },
  'sentence_195': {
    'answer': 'Thomas got three dark windows',
    'testVideos':
        ['Structured Sentences/F1/Thomas got three dark windows_1.mp4'],
  },
  'sentence_196': {
    'answer': 'Doris kept twelve small houses',
    'testVideos':
        ['Structured Sentences/F1/Doris kept twelve small houses_1.mp4'],
  },
  'sentence_197': {
    'answer': 'Nina has fifteen white spoons',
    'testVideos':
        ['Structured Sentences/F1/Nina has fifteen white spoons_1.mp4'],
  },
  'sentence_198': {
    'answer': 'Nina sold twelve old windows',
    'testVideos':
        ['Structured Sentences/F1/Nina sold twelve old windows_1.mp4'],
  },
  'sentence_199': {
    'answer': 'Peter bought fifteen green rings',
    'testVideos':
        ['Structured Sentences/F1/Peter bought fifteen green rings_1.mp4'],
  },
  'sentence_200': {
    'answer': 'Peter got nineteen heavy windows',
    'testVideos':
        ['Structured Sentences/F1/Peter got nineteen heavy windows_1.mp4'],
  },
  'sentence_201': {
    'answer': 'Nina sold nine heavy desks',
    'testVideos': ['Structured Sentences/F1/Nina sold nine heavy desks_1.mp4'],
  },
  'sentence_202': {
    'answer': 'Kathy sold sixty cheap tables',
    'testVideos':
        ['Structured Sentences/F1/Kathy sold sixty cheap tables_1.mp4'],
  },
  'sentence_203': {
    'answer': 'Peter ordered nine heavy desks',
    'testVideos':
        ['Structured Sentences/F1/Peter ordered nine heavy desks_1.mp4'],
  },
  'sentence_204': {
    'answer': 'Peter got two cheap houses',
    'testVideos': ['Structured Sentences/F1/Peter got two cheap houses_1.mp4'],
  },
  'sentence_205': {
    'answer': 'Nina got twelve white toys',
    'testVideos': ['Structured Sentences/F1/Nina got twelve white toys_1.mp4'],
  },
  'sentence_206': {
    'answer': 'Kathy got sixty pretty desks',
    'testVideos':
        ['Structured Sentences/F1/Kathy got sixty pretty desks_1.mp4'],
  },
  'sentence_207': {
    'answer': 'Steven wants fifteen heavy tables',
    'testVideos':
        ['Structured Sentences/F1/Steven wants fifteen heavy tables_1.mp4'],
  },
  'sentence_208': {
    'answer': 'Thomas has sixty dark houses',
    'testVideos':
        ['Structured Sentences/F1/Thomas has sixty dark houses_1.mp4'],
  },
  'sentence_209': {
    'answer': 'William prefers three large desks',
    'testVideos':
        ['Structured Sentences/F1/William prefers three large desks_1.mp4'],
  },
  'sentence_210': {
    'answer': 'William sees sixty cheap chairs',
    'testVideos':
        ['Structured Sentences/F1/William sees sixty cheap chairs_1.mp4'],
  },
  'sentence_211': {
    'answer': 'Doris prefers nineteen dark flowers',
    'testVideos':
        ['Structured Sentences/F1/Doris prefers nineteen dark flowers_1.mp4'],
  },
  'sentence_212': {
    'answer': 'Peter bought nineteen dark rings',
    'testVideos':
        ['Structured Sentences/F1/Peter bought nineteen dark rings_1.mp4'],
  },
  'sentence_213': {
    'answer': 'Kathy ordered three dark desks',
    'testVideos':
        ['Structured Sentences/F1/Kathy ordered three dark desks_1.mp4'],
  },
  'sentence_214': {
    'answer': 'Peter has seven large rings',
    'testVideos': ['Structured Sentences/F1/Peter has seven large rings_1.mp4'],
  },
  'sentence_215': {
    'answer': 'Steven sees nineteen dark sofas',
    'testVideos':
        ['Structured Sentences/F1/Steven sees nineteen dark sofas_1.mp4'],
  },
  'sentence_216': {
    'answer': 'Peter got four pretty houses',
    'testVideos':
        ['Structured Sentences/F1/Peter got four pretty houses_1.mp4'],
  },
  'sentence_217': {
    'answer': 'Thomas sees seven green houses',
    'testVideos':
        ['Structured Sentences/F1/Thomas sees seven green houses_1.mp4'],
  },
  'sentence_218': {
    'answer': 'Peter wants nineteen red windows',
    'testVideos':
        ['Structured Sentences/F1/Peter wants nineteen red windows_1.mp4'],
  },
  'sentence_219': {
    'answer': 'Thomas wants fifteen heavy tables',
    'testVideos':
        ['Structured Sentences/F1/Thomas wants fifteen heavy tables_1.mp4'],
  },
  'sentence_220': {
    'answer': 'William gives nine green houses',
    'testVideos':
        ['Structured Sentences/F1/William gives nine green houses_1.mp4'],
  },
  'sentence_221': {
    'answer': 'Rachel got eight white desks',
    'testVideos':
        ['Structured Sentences/F1/Rachel got eight white desks_1.mp4'],
  },
  'sentence_222': {
    'answer': 'Peter gives fifteen small spoons',
    'testVideos':
        ['Structured Sentences/F1/Peter gives fifteen small spoons_1.mp4'],
  },
  'sentence_223': {
    'answer': 'Allen bought seven green desks',
    'testVideos':
        ['Structured Sentences/F1/Allen bought seven green desks_1.mp4'],
  },
  'sentence_224': {
    'answer': 'Steven got twelve pretty windows',
    'testVideos':
        ['Structured Sentences/F1/Steven got twelve pretty windows_1.mp4'],
  },
  'sentence_225': {
    'answer': 'Doris gives two large sofas',
    'testVideos': ['Structured Sentences/F1/Doris gives two large sofas_1.mp4'],
  },
  'sentence_226': {
    'answer': 'Doris kept seven large sofas',
    'testVideos':
        ['Structured Sentences/F1/Doris kept seven large sofas_1.mp4'],
  },
  'sentence_227': {
    'answer': 'Nina sold four heavy rings',
    'testVideos': ['Structured Sentences/F1/Nina sold four heavy rings_1.mp4'],
  },
  'sentence_228': {
    'answer': 'Lucy gives eight cheap toys',
    'testVideos': ['Structured Sentences/F1/Lucy gives eight cheap toys_1.mp4'],
  },
  'sentence_229': {
    'answer': 'Peter has fifteen dark flowers',
    'testVideos':
        ['Structured Sentences/F1/Peter has fifteen dark flowers_1.mp4'],
  },
  'sentence_230': {
    'answer': 'Allen bought seven pretty tables',
    'testVideos':
        ['Structured Sentences/F1/Allen bought seven pretty tables_1.mp4'],
  },
  'sentence_231': {
    'answer': 'William prefers three green flowers',
    'testVideos':
        ['Structured Sentences/F1/William prefers three green flowers_1.mp4'],
  },
  'sentence_232': {
    'answer': 'Kathy has eight red flowers',
    'testVideos': ['Structured Sentences/F1/Kathy has eight red flowers_1.mp4'],
  },
  'sentence_233': {
    'answer': 'Steven has four heavy chairs',
    'testVideos':
        ['Structured Sentences/F1/Steven has four heavy chairs_1.mp4'],
  },
  'sentence_234': {
    'answer': 'Nina prefers three red spoons',
    'testVideos':
        ['Structured Sentences/F1/Nina prefers three red spoons_1.mp4'],
  },
  'sentence_235': {
    'answer': 'Lucy wants fifteen small sofas',
    'testVideos':
        ['Structured Sentences/F1/Lucy wants fifteen small sofas_1.mp4'],
  },
  'sentence_236': {
    'answer': 'Lucy prefers three dark houses',
    'testVideos':
        ['Structured Sentences/F1/Lucy prefers three dark houses_1.mp4'],
  },
  'sentence_237': {
    'answer': 'Doris got eight large rings',
    'testVideos': ['Structured Sentences/F1/Doris got eight large rings_1.mp4'],
  },
  'sentence_238': {
    'answer': 'Kathy bought eight green flowers',
    'testVideos':
        ['Structured Sentences/F1/Kathy bought eight green flowers_1.mp4'],
  },
  'sentence_239': {
    'answer': 'Peter has seven white flowers',
    'testVideos':
        ['Structured Sentences/F1/Peter has seven white flowers_1.mp4'],
  },
  'sentence_240': {
    'answer': 'Thomas gives three white windows',
    'testVideos':
        ['Structured Sentences/F1/Thomas gives three white windows_1.mp4'],
  },
  'sentence_241': {
    'answer': 'Doris got three old toys',
    'testVideos': ['Structured Sentences/F1/Doris got three old toys_1.mp4'],
  },
  'sentence_242': {
    'answer': 'William wants three pretty desks',
    'testVideos':
        ['Structured Sentences/F1/William wants three pretty desks_1.mp4'],
  },
  'sentence_243': {
    'answer': 'Allen gives four pretty desks',
    'testVideos':
        ['Structured Sentences/F1/Allen gives four pretty desks_1.mp4'],
  },
  'sentence_244': {
    'answer': 'Rachel sold fifteen old rings',
    'testVideos':
        ['Structured Sentences/F1/Rachel sold fifteen old rings_1.mp4'],
  },
  'sentence_245': {
    'answer': 'Kathy wants fifteen green windows',
    'testVideos':
        ['Structured Sentences/F1/Kathy wants fifteen green windows_1.mp4'],
  },
  'sentence_246': {
    'answer': 'Doris wants twelve cheap toys',
    'testVideos':
        ['Structured Sentences/F1/Doris wants twelve cheap toys_1.mp4'],
  },
  'sentence_247': {
    'answer': 'Peter ordered sixty white desks',
    'testVideos':
        ['Structured Sentences/F1/Peter ordered sixty white desks_1.mp4'],
  },
  'sentence_248': {
    'answer': 'Doris got seven cheap desks',
    'testVideos': ['Structured Sentences/F1/Doris got seven cheap desks_1.mp4'],
  },
  'sentence_249': {
    'answer': 'Rachel sold seven old tables',
    'testVideos':
        ['Structured Sentences/F1/Rachel sold seven old tables_1.mp4'],
  },
  'sentence_250': {
    'answer': 'Lucy sold nine large spoons',
    'testVideos': ['Structured Sentences/F1/Lucy sold nine large spoons_1.mp4'],
  },
  'sentence_251': {
    'answer': 'Thomas prefers nineteen white toys',
    'testVideos':
        ['Structured Sentences/F1/Thomas prefers nineteen white toys_1.mp4'],
  },
  'sentence_252': {
    'answer': 'Steven gives fifteen white toys',
    'testVideos':
        ['Structured Sentences/F1/Steven gives fifteen white toys_1.mp4'],
  },
  'sentence_253': {
    'answer': 'Allen bought nineteen large chairs',
    'testVideos':
        ['Structured Sentences/F1/Allen bought nineteen large chairs_1.mp4'],
  },
  'sentence_254': {
    'answer': 'Peter sees sixty old sofas',
    'testVideos': ['Structured Sentences/F1/Peter sees sixty old sofas_1.mp4'],
  },
  'sentence_255': {
    'answer': 'Doris got four red toys',
    'testVideos': ['Structured Sentences/F1/Doris got four red toys_1.mp4'],
  },
  'sentence_256': {
    'answer': 'William gives twelve red rings',
    'testVideos':
        ['Structured Sentences/F1/William gives twelve red rings_1.mp4'],
  },
  'sentence_257': {
    'answer': 'Kathy kept two dark houses',
    'testVideos': ['Structured Sentences/F1/Kathy kept two dark houses_1.mp4'],
  },
  'sentence_258': {
    'answer': 'Lucy got four heavy toys',
    'testVideos': ['Structured Sentences/F1/Lucy got four heavy toys_1.mp4'],
  },
  'sentence_259': {
    'answer': 'Steven sold twelve large toys',
    'testVideos':
        ['Structured Sentences/F1/Steven sold twelve large toys_1.mp4'],
  },
  'sentence_260': {
    'answer': 'Allen ordered three dark spoons',
    'testVideos':
        ['Structured Sentences/F1/Allen ordered three dark spoons_1.mp4'],
  },
  'sentence_261': {
    'answer': 'Rachel got three white desks',
    'testVideos':
        ['Structured Sentences/F1/Rachel got three white desks_1.mp4'],
  },
  'sentence_262': {
    'answer': 'Peter wants twelve small tables',
    'testVideos':
        ['Structured Sentences/F1/Peter wants twelve small tables_1.mp4'],
  },
  'sentence_263': {
    'answer': 'Thomas has four pretty rings',
    'testVideos':
        ['Structured Sentences/F1/Thomas has four pretty rings_1.mp4'],
  },
  'sentence_264': {
    'answer': 'William kept nine large rings',
    'testVideos':
        ['Structured Sentences/F1/William kept nine large rings_1.mp4'],
  },
  'sentence_265': {
    'answer': 'Peter has twelve old flowers',
    'testVideos':
        ['Structured Sentences/F1/Peter has twelve old flowers_1.mp4'],
  },
  'sentence_266': {
    'answer': 'Nina has eight green spoons',
    'testVideos': ['Structured Sentences/F1/Nina has eight green spoons_1.mp4'],
  },
  'sentence_267': {
    'answer': 'William ordered nine dark windows',
    'testVideos':
        ['Structured Sentences/F1/William ordered nine dark windows_1.mp4'],
  },
  'sentence_268': {
    'answer': 'Kathy has fifteen heavy desks',
    'testVideos':
        ['Structured Sentences/F1/Kathy has fifteen heavy desks_1.mp4'],
  },
  'sentence_269': {
    'answer': 'William gives nineteen small sofas',
    'testVideos':
        ['Structured Sentences/F1/William gives nineteen small sofas_1.mp4'],
  },
  'sentence_270': {
    'answer': 'Lucy ordered twelve dark houses',
    'testVideos':
        ['Structured Sentences/F1/Lucy ordered twelve dark houses_1.mp4'],
  },
  'sentence_271': {
    'answer': 'Peter wants twelve large windows',
    'testVideos':
        ['Structured Sentences/F1/Peter wants twelve large windows_1.mp4'],
  },
  'sentence_272': {
    'answer': 'Peter got three cheap tables',
    'testVideos':
        ['Structured Sentences/F1/Peter got three cheap tables_1.mp4'],
  },
  'sentence_273': {
    'answer': 'Peter got twelve old desks',
    'testVideos': ['Structured Sentences/F1/Peter got twelve old desks_1.mp4'],
  },
  'sentence_274': {
    'answer': 'Kathy gives three small desks',
    'testVideos':
        ['Structured Sentences/F1/Kathy gives three small desks_1.mp4'],
  },
  'sentence_275': {
    'answer': 'Steven kept fifteen small sofas',
    'testVideos':
        ['Structured Sentences/F1/Steven kept fifteen small sofas_1.mp4'],
  },
  'sentence_276': {
    'answer': 'Peter ordered eight green flowers',
    'testVideos':
        ['Structured Sentences/F1/Peter ordered eight green flowers_1.mp4'],
  },
  'sentence_277': {
    'answer': 'Thomas sold two old chairs',
    'testVideos': ['Structured Sentences/F1/Thomas sold two old chairs_1.mp4'],
  },
  'sentence_278': {
    'answer': 'Thomas sees eight old toys',
    'testVideos': ['Structured Sentences/F1/Thomas sees eight old toys_1.mp4'],
  },
  'sentence_279': {
    'answer': 'Thomas has two pretty flowers',
    'testVideos':
        ['Structured Sentences/F1/Thomas has two pretty flowers_1.mp4'],
  },
  'sentence_280': {
    'answer': 'Thomas sold seven old desks',
    'testVideos': ['Structured Sentences/F1/Thomas sold seven old desks_1.mp4'],
  },
  'sentence_281': {
    'answer': 'Steven got three large chairs',
    'testVideos':
        ['Structured Sentences/F1/Steven got three large chairs_1.mp4'],
  },
  'sentence_282': {
    'answer': 'Rachel bought eight white rings',
    'testVideos':
        ['Structured Sentences/F1/Rachel bought eight white rings_1.mp4'],
  },
  'sentence_283': {
    'answer': 'Allen prefers two red tables',
    'testVideos':
        ['Structured Sentences/F1/Allen prefers two red tables_1.mp4'],
  },
  'sentence_284': {
    'answer': 'Rachel bought eight dark windows',
    'testVideos':
        ['Structured Sentences/F1/Rachel bought eight dark windows_1.mp4'],
  },
  'sentence_285': {
    'answer': 'William sold three old windows',
    'testVideos':
        ['Structured Sentences/F1/William sold three old windows_1.mp4'],
  },
  'sentence_286': {
    'answer': 'Lucy wants nine large toys',
    'testVideos': ['Structured Sentences/F1/Lucy wants nine large toys_1.mp4'],
  },
  'sentence_287': {
    'answer': 'Thomas got eight red windows',
    'testVideos':
        ['Structured Sentences/F1/Thomas got eight red windows_1.mp4'],
  },
  'sentence_288': {
    'answer': 'Doris sees twelve red toys',
    'testVideos': ['Structured Sentences/F1/Doris sees twelve red toys_1.mp4'],
  },
  'sentence_289': {
    'answer': 'Kathy gives eight large spoons',
    'testVideos':
        ['Structured Sentences/F1/Kathy gives eight large spoons_1.mp4'],
  },
  'sentence_290': {
    'answer': 'Thomas sold fifteen white spoons',
    'testVideos':
        ['Structured Sentences/F1/Thomas sold fifteen white spoons_1.mp4'],
  },
  'sentence_291': {
    'answer': 'Peter kept two red desks',
    'testVideos': ['Structured Sentences/F1/Peter kept two red desks_1.mp4'],
  },
  'sentence_292': {
    'answer': 'Doris sold nine dark tables',
    'testVideos': ['Structured Sentences/F1/Doris sold nine dark tables_1.mp4'],
  },
  'sentence_293': {
    'answer': 'Doris has sixty dark spoons',
    'testVideos': ['Structured Sentences/F1/Doris has sixty dark spoons_1.mp4'],
  },
  'sentence_294': {
    'answer': 'Thomas gives twelve heavy toys',
    'testVideos':
        ['Structured Sentences/F1/Thomas gives twelve heavy toys_1.mp4'],
  },
  'sentence_295': {
    'answer': 'Thomas got twelve heavy windows',
    'testVideos':
        ['Structured Sentences/F1/Thomas got twelve heavy windows_1.mp4'],
  },
  'sentence_296': {
    'answer': 'Peter got sixty small chairs',
    'testVideos':
        ['Structured Sentences/F1/Peter got sixty small chairs_1.mp4'],
  },
  'sentence_297': {
    'answer': 'Steven ordered fifteen white spoons',
    'testVideos':
        ['Structured Sentences/F1/Steven ordered fifteen white spoons_1.mp4'],
  },
  'sentence_298': {
    'answer': 'Kathy kept eight heavy desks',
    'testVideos':
        ['Structured Sentences/F1/Kathy kept eight heavy desks_1.mp4'],
  },
  'sentence_299': {
    'answer': 'Peter got three pretty flowers',
    'testVideos':
        ['Structured Sentences/F1/Peter got three pretty flowers_1.mp4'],
  },
  'sentence_300': {
    'answer': 'Steven got nine large windows',
    'testVideos':
        ['Structured Sentences/F1/Steven got nine large windows_1.mp4'],
  },
  'sentence_301': {
    'answer': 'Nina gives two pretty rings',
    'testVideos': ['Structured Sentences/F1/Nina gives two pretty rings_1.mp4'],
  },
  'sentence_302': {
    'answer': 'Thomas bought three red windows',
    'testVideos':
        ['Structured Sentences/F1/Thomas bought three red windows_1.mp4'],
  },
  'sentence_303': {
    'answer': 'Allen prefers two heavy desks',
    'testVideos':
        ['Structured Sentences/F1/Allen prefers two heavy desks_1.mp4'],
  },
  'sentence_304': {
    'answer': 'Allen kept twelve white tables',
    'testVideos':
        ['Structured Sentences/F1/Allen kept twelve white tables_1.mp4'],
  },
  'sentence_305': {
    'answer': 'Steven bought fifteen large spoons',
    'testVideos':
        ['Structured Sentences/F1/Steven bought fifteen large spoons_1.mp4'],
  },
  'sentence_306': {
    'answer': 'Peter has seven pretty desks',
    'testVideos':
        ['Structured Sentences/F1/Peter has seven pretty desks_1.mp4'],
  },
  'sentence_307': {
    'answer': 'Rachel wants twelve white sofas',
    'testVideos':
        ['Structured Sentences/F1/Rachel wants twelve white sofas_1.mp4'],
  },
  'sentence_308': {
    'answer': 'Allen gives fifteen old desks',
    'testVideos':
        ['Structured Sentences/F1/Allen gives fifteen old desks_1.mp4'],
  },
  'sentence_309': {
    'answer': 'Lucy gives eight white desks',
    'testVideos':
        ['Structured Sentences/F1/Lucy gives eight white desks_1.mp4'],
  },
  'sentence_310': {
    'answer': 'Thomas wants three cheap spoons',
    'testVideos':
        ['Structured Sentences/F1/Thomas wants three cheap spoons_1.mp4'],
  },
  'sentence_311': {
    'answer': 'Steven sold fifteen pretty houses',
    'testVideos':
        ['Structured Sentences/F1/Steven sold fifteen pretty houses_1.mp4'],
  },
  'sentence_312': {
    'answer': 'Steven wants three cheap desks',
    'testVideos':
        ['Structured Sentences/F1/Steven wants three cheap desks_1.mp4'],
  },
  'sentence_313': {
    'answer': 'Doris has three dark desks',
    'testVideos': ['Structured Sentences/F1/Doris has three dark desks_1.mp4'],
  },
  'sentence_314': {
    'answer': 'Peter gives fifteen large desks',
    'testVideos':
        ['Structured Sentences/F1/Peter gives fifteen large desks_1.mp4'],
  },
  'sentence_315': {
    'answer': 'Allen wants three pretty chairs',
    'testVideos':
        ['Structured Sentences/F1/Allen wants three pretty chairs_1.mp4'],
  },
  'sentence_316': {
    'answer': 'Steven ordered seven small houses',
    'testVideos':
        ['Structured Sentences/F1/Steven ordered seven small houses_1.mp4'],
  },
  'sentence_317': {
    'answer': 'Peter bought four large windows',
    'testVideos':
        ['Structured Sentences/F1/Peter bought four large windows_1.mp4'],
  },
  'sentence_318': {
    'answer': 'Lucy has seven green toys',
    'testVideos': ['Structured Sentences/F1/Lucy has seven green toys_1.mp4'],
  },
  'sentence_319': {
    'answer': 'Rachel sold seven white tables',
    'testVideos':
        ['Structured Sentences/F1/Rachel sold seven white tables_1.mp4'],
  },
  'sentence_320': {
    'answer': 'Doris got eight heavy spoons',
    'testVideos':
        ['Structured Sentences/F1/Doris got eight heavy spoons_1.mp4'],
  },
  'sentence_321': {
    'answer': 'Lucy sold twelve cheap spoons',
    'testVideos':
        ['Structured Sentences/F1/Lucy sold twelve cheap spoons_1.mp4'],
  },
  'sentence_322': {
    'answer': 'William bought four red tables',
    'testVideos':
        ['Structured Sentences/F1/William bought four red tables_1.mp4'],
  },
  'sentence_323': {
    'answer': 'Allen ordered twelve heavy rings',
    'testVideos':
        ['Structured Sentences/F1/Allen ordered twelve heavy rings_1.mp4'],
  },
  'sentence_324': {
    'answer': 'Nina sees nineteen old chairs',
    'testVideos':
        ['Structured Sentences/F1/Nina sees nineteen old chairs_1.mp4'],
  },
  'sentence_325': {
    'answer': 'Peter prefers four large rings',
    'testVideos':
        ['Structured Sentences/F1/Peter prefers four large rings_1.mp4'],
  },
  'sentence_326': {
    'answer': 'Peter ordered nine old windows',
    'testVideos':
        ['Structured Sentences/F1/Peter ordered nine old windows_1.mp4'],
  },
  'sentence_327': {
    'answer': 'Rachel got two red toys',
    'testVideos': ['Structured Sentences/F1/Rachel got two red toys_1.mp4'],
  },
  'sentence_328': {
    'answer': 'Peter ordered fifteen green tables',
    'testVideos':
        ['Structured Sentences/F1/Peter ordered fifteen green tables_1.mp4'],
  },
  'sentence_329': {
    'answer': 'Peter sees fifteen green chairs',
    'testVideos':
        ['Structured Sentences/F1/Peter sees fifteen green chairs_1.mp4'],
  },
  'sentence_330': {
    'answer': 'Peter prefers eight cheap chairs',
    'testVideos':
        ['Structured Sentences/F1/Peter prefers eight cheap chairs_1.mp4'],
  },
  'sentence_331': {
    'answer': 'Thomas got twelve small toys',
    'testVideos':
        ['Structured Sentences/F1/Thomas got twelve small toys_1.mp4'],
  },
  'sentence_332': {
    'answer': 'Kathy gives nineteen red spoons',
    'testVideos':
        ['Structured Sentences/F1/Kathy gives nineteen red spoons_1.mp4'],
  },
  'sentence_333': {
    'answer': 'Nina got three old houses',
    'testVideos': ['Structured Sentences/F1/Nina got three old houses_1.mp4'],
  },
  'sentence_334': {
    'answer': 'William ordered two old desks',
    'testVideos':
        ['Structured Sentences/F1/William ordered two old desks_1.mp4'],
  },
  'sentence_335': {
    'answer': 'Thomas ordered two dark desks',
    'testVideos':
        ['Structured Sentences/F1/Thomas ordered two dark desks_1.mp4'],
  },
  'sentence_336': {
    'answer': 'Rachel prefers fifteen large houses',
    'testVideos':
        ['Structured Sentences/F1/Rachel prefers fifteen large houses_1.mp4'],
  },
  'sentence_337': {
    'answer': 'Rachel gives eight dark spoons',
    'testVideos':
        ['Structured Sentences/F1/Rachel gives eight dark spoons_1.mp4'],
  },
  'sentence_338': {
    'answer': 'Rachel got nineteen white spoons',
    'testVideos':
        ['Structured Sentences/F1/Rachel got nineteen white spoons_1.mp4'],
  },
  'sentence_339': {
    'answer': 'Kathy got nine dark desks',
    'testVideos': ['Structured Sentences/F1/Kathy got nine dark desks_1.mp4'],
  },
  'sentence_340': {
    'answer': 'Lucy ordered three red spoons',
    'testVideos':
        ['Structured Sentences/F1/Lucy ordered three red spoons_1.mp4'],
  },
  'sentence_341': {
    'answer': 'Nina wants three heavy windows',
    'testVideos':
        ['Structured Sentences/F1/Nina wants three heavy windows_1.mp4'],
  },
  'sentence_342': {
    'answer': 'Thomas got seven large desks',
    'testVideos':
        ['Structured Sentences/F1/Thomas got seven large desks_1.mp4'],
  },
  'sentence_343': {
    'answer': 'Lucy wants three cheap flowers',
    'testVideos':
        ['Structured Sentences/F1/Lucy wants three cheap flowers_1.mp4'],
  },
  'sentence_344': {
    'answer': 'Steven sold eight cheap houses',
    'testVideos':
        ['Structured Sentences/F1/Steven sold eight cheap houses_1.mp4'],
  },
  'sentence_345': {
    'answer': 'Peter got nineteen large desks',
    'testVideos':
        ['Structured Sentences/F1/Peter got nineteen large desks_1.mp4'],
  },
  'sentence_346': {
    'answer': 'Peter prefers four green desks',
    'testVideos':
        ['Structured Sentences/F1/Peter prefers four green desks_1.mp4'],
  },
  'sentence_347': {
    'answer': 'Nina sold fifteen large desks',
    'testVideos':
        ['Structured Sentences/F1/Nina sold fifteen large desks_1.mp4'],
  },
  'sentence_348': {
    'answer': 'Thomas got twelve large toys',
    'testVideos':
        ['Structured Sentences/F1/Thomas got twelve large toys_1.mp4'],
  },
  'sentence_349': {
    'answer': 'Rachel kept eight large sofas',
    'testVideos':
        ['Structured Sentences/F1/Rachel kept eight large sofas_1.mp4'],
  },
  'sentence_350': {
    'answer': 'Rachel got eight small spoons',
    'testVideos':
        ['Structured Sentences/F1/Rachel got eight small spoons_1.mp4'],
  },
  'sentence_351': {
    'answer': 'Steven ordered seven cheap windows',
    'testVideos':
        ['Structured Sentences/F1/Steven ordered seven cheap windows_1.mp4'],
  },
  'sentence_352': {
    'answer': 'Steven prefers fifteen red spoons',
    'testVideos':
        ['Structured Sentences/F1/Steven prefers fifteen red spoons_1.mp4'],
  },
  'sentence_353': {
    'answer': 'Peter got sixty red spoons',
    'testVideos': ['Structured Sentences/F1/Peter got sixty red spoons_1.mp4'],
  },
  'sentence_354': {
    'answer': 'Kathy prefers sixty cheap desks',
    'testVideos':
        ['Structured Sentences/F1/Kathy prefers sixty cheap desks_1.mp4'],
  },
  'sentence_355': {
    'answer': 'Nina has eight small windows',
    'testVideos':
        ['Structured Sentences/F1/Nina has eight small windows_1.mp4'],
  },
  'sentence_356': {
    'answer': 'Thomas has three white desks',
    'testVideos':
        ['Structured Sentences/F1/Thomas has three white desks_1.mp4'],
  },
  'sentence_357': {
    'answer': 'Steven sold three red desks',
    'testVideos': ['Structured Sentences/F1/Steven sold three red desks_1.mp4'],
  },
  'sentence_358': {
    'answer': 'Rachel wants fifteen red windows',
    'testVideos':
        ['Structured Sentences/F1/Rachel wants fifteen red windows_1.mp4'],
  },
  'sentence_359': {
    'answer': 'Peter gives twelve old toys',
    'testVideos': ['Structured Sentences/F1/Peter gives twelve old toys_1.mp4'],
  },
  'sentence_360': {
    'answer': 'Kathy gives nine green rings',
    'testVideos':
        ['Structured Sentences/F1/Kathy gives nine green rings_1.mp4'],
  },
  'sentence_361': {
    'answer': 'Thomas sees nineteen white desks',
    'testVideos':
        ['Structured Sentences/F1/Thomas sees nineteen white desks_1.mp4'],
  },
  'sentence_362': {
    'answer': 'Peter gives three small desks',
    'testVideos':
        ['Structured Sentences/F1/Peter gives three small desks_1.mp4'],
  },
  'sentence_363': {
    'answer': 'Thomas kept fifteen dark windows',
    'testVideos':
        ['Structured Sentences/F1/Thomas kept fifteen dark windows_1.mp4'],
  },
  'sentence_364': {
    'answer': 'Allen prefers four small flowers',
    'testVideos':
        ['Structured Sentences/F1/Allen prefers four small flowers_1.mp4'],
  },
  'sentence_365': {
    'answer': 'Nina sold fifteen old flowers',
    'testVideos':
        ['Structured Sentences/F1/Nina sold fifteen old flowers_1.mp4'],
  },
  'sentence_366': {
    'answer': 'Allen sold nine heavy tables',
    'testVideos':
        ['Structured Sentences/F1/Allen sold nine heavy tables_1.mp4'],
  },
  'sentence_367': {
    'answer': 'Peter sold eight large sofas',
    'testVideos':
        ['Structured Sentences/F1/Peter sold eight large sofas_1.mp4'],
  },
  'sentence_368': {
    'answer': 'Thomas got nine pretty sofas',
    'testVideos':
        ['Structured Sentences/F1/Thomas got nine pretty sofas_1.mp4'],
  },
  'sentence_369': {
    'answer': 'Doris ordered seven red rings',
    'testVideos':
        ['Structured Sentences/F1/Doris ordered seven red rings_1.mp4'],
  },
  'sentence_370': {
    'answer': 'Kathy ordered two green desks',
    'testVideos':
        ['Structured Sentences/F1/Kathy ordered two green desks_1.mp4'],
  },
  'sentence_371': {
    'answer': 'Lucy has nineteen white chairs',
    'testVideos':
        ['Structured Sentences/F1/Lucy has nineteen white chairs_1.mp4'],
  },
  'sentence_372': {
    'answer': 'Rachel sold seven green houses',
    'testVideos':
        ['Structured Sentences/F1/Rachel sold seven green houses_1.mp4'],
  },
  'sentence_373': {
    'answer': 'Nina ordered nine white houses',
    'testVideos':
        ['Structured Sentences/F1/Nina ordered nine white houses_1.mp4'],
  },
  'sentence_374': {
    'answer': 'Doris sold nineteen green desks',
    'testVideos':
        ['Structured Sentences/F1/Doris sold nineteen green desks_1.mp4'],
  },
  'sentence_375': {
    'answer': 'Lucy sold fifteen cheap spoons',
    'testVideos':
        ['Structured Sentences/F1/Lucy sold fifteen cheap spoons_1.mp4'],
  },
  'sentence_376': {
    'answer': 'Peter kept eight dark toys',
    'testVideos': ['Structured Sentences/F1/Peter kept eight dark toys_1.mp4'],
  },
  'sentence_377': {
    'answer': 'Nina wants nine cheap flowers',
    'testVideos':
        ['Structured Sentences/F1/Nina wants nine cheap flowers_1.mp4'],
  },
  'sentence_378': {
    'answer': 'William bought two heavy windows',
    'testVideos':
        ['Structured Sentences/F1/William bought two heavy windows_1.mp4'],
  },
  'sentence_379': {
    'answer': 'Steven sees three pretty desks',
    'testVideos':
        ['Structured Sentences/F1/Steven sees three pretty desks_1.mp4'],
  },
  'sentence_380': {
    'answer': 'Peter ordered two dark sofas',
    'testVideos':
        ['Structured Sentences/F1/Peter ordered two dark sofas_1.mp4'],
  },
  'sentence_381': {
    'answer': 'Allen kept seven red houses',
    'testVideos': ['Structured Sentences/F1/Allen kept seven red houses_1.mp4'],
  },
  'sentence_382': {
    'answer': 'Peter gives sixty large tables',
    'testVideos':
        ['Structured Sentences/F1/Peter gives sixty large tables_1.mp4'],
  },
  'sentence_383': {
    'answer': 'Peter wants three green spoons',
    'testVideos':
        ['Structured Sentences/F1/Peter wants three green spoons_1.mp4'],
  },
  'sentence_384': {
    'answer': 'Rachel got eight small houses',
    'testVideos':
        ['Structured Sentences/F1/Rachel got eight small houses_1.mp4'],
  },
  'sentence_385': {
    'answer': 'Thomas ordered three old houses',
    'testVideos':
        ['Structured Sentences/F1/Thomas ordered three old houses_1.mp4'],
  },
  'sentence_386': {
    'answer': 'Lucy has three red desks',
    'testVideos': ['Structured Sentences/F1/Lucy has three red desks_1.mp4'],
  },
  'sentence_387': {
    'answer': 'Allen got sixty old desks',
    'testVideos': ['Structured Sentences/F1/Allen got sixty old desks_1.mp4'],
  },
  'sentence_388': {
    'answer': 'William wants sixty pretty chairs',
    'testVideos':
        ['Structured Sentences/F1/William wants sixty pretty chairs_1.mp4'],
  },
  'sentence_389': {
    'answer': 'Lucy kept two old rings',
    'testVideos': ['Structured Sentences/F1/Lucy kept two old rings_1.mp4'],
  },
  'sentence_390': {
    'answer': 'William got three small toys',
    'testVideos':
        ['Structured Sentences/F1/William got three small toys_1.mp4'],
  },
  'sentence_391': {
    'answer': 'Thomas prefers nineteen small toys',
    'testVideos':
        ['Structured Sentences/F1/Thomas prefers nineteen small toys_1.mp4'],
  },
  'sentence_392': {
    'answer': 'Nina sold nineteen dark windows',
    'testVideos':
        ['Structured Sentences/F1/Nina sold nineteen dark windows_1.mp4'],
  },
  'sentence_393': {
    'answer': 'William kept two red windows',
    'testVideos':
        ['Structured Sentences/F1/William kept two red windows_1.mp4'],
  },
  'sentence_394': {
    'answer': 'Lucy bought twelve green desks',
    'testVideos':
        ['Structured Sentences/F1/Lucy bought twelve green desks_1.mp4'],
  },
  'sentence_395': {
    'answer': 'Lucy got fifteen cheap chairs',
    'testVideos':
        ['Structured Sentences/F1/Lucy got fifteen cheap chairs_1.mp4'],
  },
  'sentence_396': {
    'answer': 'Steven prefers three pretty desks',
    'testVideos':
        ['Structured Sentences/F1/Steven prefers three pretty desks_1.mp4'],
  },
  'sentence_397': {
    'answer': 'Thomas gives sixty green tables',
    'testVideos':
        ['Structured Sentences/F1/Thomas gives sixty green tables_1.mp4'],
  },
  'sentence_398': {
    'answer': 'Thomas ordered two small tables',
    'testVideos':
        ['Structured Sentences/F1/Thomas ordered two small tables_1.mp4'],
  },
  'sentence_399': {
    'answer': 'Steven bought twelve white chairs',
    'testVideos':
        ['Structured Sentences/F1/Steven bought twelve white chairs_1.mp4'],
  },
  'sentence_400': {
    'answer': 'Rachel wants twelve white flowers',
    'testVideos':
        ['Structured Sentences/F1/Rachel wants twelve white flowers_1.mp4'],
  },
  'sentence_401': {
    'answer': 'Nina wants three heavy flowers',
    'testVideos':
        ['Structured Sentences/F1/Nina wants three heavy flowers_1.mp4'],
  },
  'sentence_402': {
    'answer': 'Peter gives twelve small desks',
    'testVideos':
        ['Structured Sentences/F1/Peter gives twelve small desks_1.mp4'],
  },
  'sentence_403': {
    'answer': 'Peter got three cheap houses',
    'testVideos':
        ['Structured Sentences/F1/Peter got three cheap houses_1.mp4'],
  },
  'sentence_404': {
    'answer': 'Kathy got seven large chairs',
    'testVideos':
        ['Structured Sentences/F1/Kathy got seven large chairs_1.mp4'],
  },
  'sentence_405': {
    'answer': 'Allen prefers four heavy flowers',
    'testVideos':
        ['Structured Sentences/F1/Allen prefers four heavy flowers_1.mp4'],
  },
  'sentence_406': {
    'answer': 'Thomas has three large desks',
    'testVideos':
        ['Structured Sentences/F1/Thomas has three large desks_1.mp4'],
  },
  'sentence_407': {
    'answer': 'William bought two green spoons',
    'testVideos':
        ['Structured Sentences/F1/William bought two green spoons_1.mp4'],
  },
  'sentence_408': {
    'answer': 'Lucy has four white chairs',
    'testVideos': ['Structured Sentences/F1/Lucy has four white chairs_1.mp4'],
  },
  'sentence_409': {
    'answer': 'Lucy got eight red flowers',
    'testVideos': ['Structured Sentences/F1/Lucy got eight red flowers_1.mp4'],
  },
  'sentence_410': {
    'answer': 'Rachel got three green houses',
    'testVideos':
        ['Structured Sentences/F1/Rachel got three green houses_1.mp4'],
  },
  'sentence_411': {
    'answer': 'Peter sold fifteen cheap chairs',
    'testVideos':
        ['Structured Sentences/F1/Peter sold fifteen cheap chairs_1.mp4'],
  },
  'sentence_412': {
    'answer': 'Allen wants eight green toys',
    'testVideos':
        ['Structured Sentences/F1/Allen wants eight green toys_1.mp4'],
  },
  'sentence_413': {
    'answer': 'Lucy has four old flowers',
    'testVideos': ['Structured Sentences/F1/Lucy has four old flowers_1.mp4'],
  },
  'sentence_414': {
    'answer': 'William sees sixty cheap flowers',
    'testVideos':
        ['Structured Sentences/F1/William sees sixty cheap flowers_1.mp4'],
  },
  'sentence_415': {
    'answer': 'Doris sold twelve old sofas',
    'testVideos': ['Structured Sentences/F1/Doris sold twelve old sofas_1.mp4'],
  },
  'sentence_416': {
    'answer': 'Thomas ordered nine heavy sofas',
    'testVideos':
        ['Structured Sentences/F1/Thomas ordered nine heavy sofas_1.mp4'],
  },
  'sentence_417': {
    'answer': 'William ordered nineteen old toys',
    'testVideos':
        ['Structured Sentences/F1/William ordered nineteen old toys_1.mp4'],
  },
  'sentence_418': {
    'answer': 'Allen sold two green chairs',
    'testVideos': ['Structured Sentences/F1/Allen sold two green chairs_1.mp4'],
  },
  'sentence_419': {
    'answer': 'Thomas got eight large windows',
    'testVideos':
        ['Structured Sentences/F1/Thomas got eight large windows_1.mp4'],
  },
  'sentence_420': {
    'answer': 'Nina kept four large tables',
    'testVideos': ['Structured Sentences/F1/Nina kept four large tables_1.mp4'],
  },
  'sentence_421': {
    'answer': 'Peter ordered three white houses',
    'testVideos':
        ['Structured Sentences/F1/Peter ordered three white houses_1.mp4'],
  },
  'sentence_422': {
    'answer': 'Lucy ordered four cheap chairs',
    'testVideos':
        ['Structured Sentences/F1/Lucy ordered four cheap chairs_1.mp4'],
  },
  'sentence_423': {
    'answer': 'Nina sees nineteen green houses',
    'testVideos':
        ['Structured Sentences/F1/Nina sees nineteen green houses_1.mp4'],
  },
  'sentence_424': {
    'answer': 'Thomas prefers fifteen small houses',
    'testVideos':
        ['Structured Sentences/F1/Thomas prefers fifteen small houses_1.mp4'],
  },
  'sentence_425': {
    'answer': 'Steven kept three old rings',
    'testVideos': ['Structured Sentences/F1/Steven kept three old rings_1.mp4'],
  },
  'sentence_426': {
    'answer': 'Nina wants two small chairs',
    'testVideos': ['Structured Sentences/F1/Nina wants two small chairs_1.mp4'],
  },
  'sentence_427': {
    'answer': 'Steven ordered sixty pretty desks',
    'testVideos':
        ['Structured Sentences/F1/Steven ordered sixty pretty desks_1.mp4'],
  },
  'sentence_428': {
    'answer': 'Allen bought nineteen small chairs',
    'testVideos':
        ['Structured Sentences/F1/Allen bought nineteen small chairs_1.mp4'],
  },
  'sentence_429': {
    'answer': 'Doris ordered three red chairs',
    'testVideos':
        ['Structured Sentences/F1/Doris ordered three red chairs_1.mp4'],
  },
  'sentence_430': {
    'answer': 'Allen kept three heavy desks',
    'testVideos':
        ['Structured Sentences/F1/Allen kept three heavy desks_1.mp4'],
  },
  'sentence_431': {
    'answer': 'Steven sold nine pretty windows',
    'testVideos':
        ['Structured Sentences/F1/Steven sold nine pretty windows_1.mp4'],
  },
  'sentence_432': {
    'answer': 'Allen got three cheap chairs',
    'testVideos':
        ['Structured Sentences/F1/Allen got three cheap chairs_1.mp4'],
  },
  'sentence_433': {
    'answer': 'Peter bought nineteen pretty rings',
    'testVideos':
        ['Structured Sentences/F1/Peter bought nineteen pretty rings_1.mp4'],
  },
  'sentence_434': {
    'answer': 'Allen ordered twelve large desks',
    'testVideos':
        ['Structured Sentences/F1/Allen ordered twelve large desks_1.mp4'],
  },
  'sentence_435': {
    'answer': 'Lucy sees eight pretty sofas',
    'testVideos':
        ['Structured Sentences/F1/Lucy sees eight pretty sofas_1.mp4'],
  },
  'sentence_436': {
    'answer': 'Peter got fifteen small desks',
    'testVideos':
        ['Structured Sentences/F1/Peter got fifteen small desks_1.mp4'],
  },
  'sentence_437': {
    'answer': 'Allen kept eight red toys',
    'testVideos': ['Structured Sentences/F1/Allen kept eight red toys_1.mp4'],
  },
  'sentence_438': {
    'answer': 'Doris ordered eight cheap rings',
    'testVideos':
        ['Structured Sentences/F1/Doris ordered eight cheap rings_1.mp4'],
  },
  'sentence_439': {
    'answer': 'William sees eight large flowers',
    'testVideos':
        ['Structured Sentences/F1/William sees eight large flowers_1.mp4'],
  },
  'sentence_440': {
    'answer': 'Lucy has two small flowers',
    'testVideos': ['Structured Sentences/F1/Lucy has two small flowers_1.mp4'],
  },
  'sentence_441': {
    'answer': 'Doris sees three old desks',
    'testVideos': ['Structured Sentences/F1/Doris sees three old desks_1.mp4'],
  },
  'sentence_442': {
    'answer': 'Lucy ordered twelve old desks',
    'testVideos':
        ['Structured Sentences/F1/Lucy ordered twelve old desks_1.mp4'],
  },
  'sentence_443': {
    'answer': 'Kathy bought nine white rings',
    'testVideos':
        ['Structured Sentences/F1/Kathy bought nine white rings_1.mp4'],
  },
  'sentence_444': {
    'answer': 'Peter ordered twelve large toys',
    'testVideos':
        ['Structured Sentences/F1/Peter ordered twelve large toys_1.mp4'],
  },
  'sentence_445': {
    'answer': 'Thomas prefers nine dark rings',
    'testVideos':
        ['Structured Sentences/F1/Thomas prefers nine dark rings_1.mp4'],
  },
  'sentence_446': {
    'answer': 'Thomas ordered nine dark flowers',
    'testVideos':
        ['Structured Sentences/F1/Thomas ordered nine dark flowers_1.mp4'],
  },
  'sentence_447': {
    'answer': 'Doris prefers three large desks',
    'testVideos':
        ['Structured Sentences/F1/Doris prefers three large desks_1.mp4'],
  },
  'sentence_448': {
    'answer': 'Steven bought four small tables',
    'testVideos':
        ['Structured Sentences/F1/Steven bought four small tables_1.mp4'],
  },
  'sentence_449': {
    'answer': 'Lucy got three green spoons',
    'testVideos': ['Structured Sentences/F1/Lucy got three green spoons_1.mp4'],
  },
  'sentence_450': {
    'answer': 'Steven prefers two old rings',
    'testVideos':
        ['Structured Sentences/F1/Steven prefers two old rings_1.mp4'],
  },
  'sentence_451': {
    'answer': 'Thomas got twelve dark houses',
    'testVideos':
        ['Structured Sentences/F1/Thomas got twelve dark houses_1.mp4'],
  },
  'sentence_452': {
    'answer': 'Kathy wants fifteen cheap desks',
    'testVideos':
        ['Structured Sentences/F1/Kathy wants fifteen cheap desks_1.mp4'],
  },
  'sentence_453': {
    'answer': 'William has seven white tables',
    'testVideos':
        ['Structured Sentences/F1/William has seven white tables_1.mp4'],
  },
  'sentence_454': {
    'answer': 'Rachel prefers two old toys',
    'testVideos': ['Structured Sentences/F1/Rachel prefers two old toys_1.mp4'],
  },
  'sentence_455': {
    'answer': 'Lucy prefers twelve large sofas',
    'testVideos':
        ['Structured Sentences/F1/Lucy prefers twelve large sofas_1.mp4'],
  },
  'sentence_456': {
    'answer': 'Rachel wants seven large chairs',
    'testVideos':
        ['Structured Sentences/F1/Rachel wants seven large chairs_1.mp4'],
  },
  'sentence_457': {
    'answer': 'Allen sold four red flowers',
    'testVideos': ['Structured Sentences/F1/Allen sold four red flowers_1.mp4'],
  },
  'sentence_458': {
    'answer': 'Rachel got four heavy spoons',
    'testVideos':
        ['Structured Sentences/F1/Rachel got four heavy spoons_1.mp4'],
  },
  'sentence_459': {
    'answer': 'Steven prefers four large houses',
    'testVideos':
        ['Structured Sentences/F1/Steven prefers four large houses_1.mp4'],
  },
  'sentence_460': {
    'answer': 'Kathy wants fifteen green tables',
    'testVideos':
        ['Structured Sentences/F1/Kathy wants fifteen green tables_1.mp4'],
  },
  'sentence_461': {
    'answer': 'Lucy gives eight small houses',
    'testVideos':
        ['Structured Sentences/F1/Lucy gives eight small houses_1.mp4'],
  },
  'sentence_462': {
    'answer': 'Rachel kept seven small rings',
    'testVideos':
        ['Structured Sentences/F1/Rachel kept seven small rings_1.mp4'],
  },
  'sentence_463': {
    'answer': 'Kathy kept seven green flowers',
    'testVideos':
        ['Structured Sentences/F1/Kathy kept seven green flowers_1.mp4'],
  },
  'sentence_464': {
    'answer': 'Rachel bought fifteen green houses',
    'testVideos':
        ['Structured Sentences/F1/Rachel bought fifteen green houses_1.mp4'],
  },
  'sentence_465': {
    'answer': 'Peter got fifteen cheap rings',
    'testVideos':
        ['Structured Sentences/F1/Peter got fifteen cheap rings_1.mp4'],
  },
  'sentence_466': {
    'answer': 'Peter wants three small tables',
    'testVideos':
        ['Structured Sentences/F1/Peter wants three small tables_1.mp4'],
  },
  'sentence_467': {
    'answer': 'Thomas sees twelve large chairs',
    'testVideos':
        ['Structured Sentences/F1/Thomas sees twelve large chairs_1.mp4'],
  },
  'sentence_468': {
    'answer': 'William ordered four heavy windows',
    'testVideos':
        ['Structured Sentences/F1/William ordered four heavy windows_1.mp4'],
  },
  'sentence_469': {
    'answer': 'Peter bought sixty pretty desks',
    'testVideos':
        ['Structured Sentences/F1/Peter bought sixty pretty desks_1.mp4'],
  },
  'sentence_470': {
    'answer': 'Lucy kept two green chairs',
    'testVideos': ['Structured Sentences/F1/Lucy kept two green chairs_1.mp4'],
  },
  'sentence_471': {
    'answer': 'Allen gives seven green flowers',
    'testVideos':
        ['Structured Sentences/F1/Allen gives seven green flowers_1.mp4'],
  },
  'sentence_472': {
    'answer': 'Peter has nineteen white windows',
    'testVideos':
        ['Structured Sentences/F1/Peter has nineteen white windows_1.mp4'],
  },
  'sentence_473': {
    'answer': 'Peter sold fifteen cheap toys',
    'testVideos':
        ['Structured Sentences/F1/Peter sold fifteen cheap toys_1.mp4'],
  },
  'sentence_474': {
    'answer': 'Peter gives three green rings',
    'testVideos':
        ['Structured Sentences/F1/Peter gives three green rings_1.mp4'],
  },
  'sentence_475': {
    'answer': 'Rachel sees three pretty chairs',
    'testVideos':
        ['Structured Sentences/F1/Rachel sees three pretty chairs_1.mp4'],
  },
  'sentence_476': {
    'answer': 'Allen bought nineteen pretty sofas',
    'testVideos':
        ['Structured Sentences/F1/Allen bought nineteen pretty sofas_1.mp4'],
  },
  'sentence_477': {
    'answer': 'Kathy sees seven heavy houses',
    'testVideos':
        ['Structured Sentences/F1/Kathy sees seven heavy houses_1.mp4'],
  },
  'sentence_478': {
    'answer': 'Doris has three red toys',
    'testVideos': ['Structured Sentences/F1/Doris has three red toys_1.mp4'],
  },
  'sentence_479': {
    'answer': 'Steven prefers seven pretty sofas',
    'testVideos':
        ['Structured Sentences/F1/Steven prefers seven pretty sofas_1.mp4'],
  },
  'sentence_480': {
    'answer': 'Lucy got fifteen small flowers',
    'testVideos':
        ['Structured Sentences/F1/Lucy got fifteen small flowers_1.mp4'],
  },
  'sentence_481': {
    'answer': 'Lucy gives fifteen white sofas',
    'testVideos':
        ['Structured Sentences/F1/Lucy gives fifteen white sofas_1.mp4'],
  },
  'sentence_482': {
    'answer': 'Doris got nine old windows',
    'testVideos': ['Structured Sentences/F1/Doris got nine old windows_1.mp4'],
  },
  'sentence_483': {
    'answer': 'Peter sold four heavy windows',
    'testVideos':
        ['Structured Sentences/F1/Peter sold four heavy windows_1.mp4'],
  },
  'sentence_484': {
    'answer': 'Nina sold two small houses',
    'testVideos': ['Structured Sentences/F1/Nina sold two small houses_1.mp4'],
  },
  'sentence_485': {
    'answer': 'William sold nine old desks',
    'testVideos': ['Structured Sentences/F1/William sold nine old desks_1.mp4'],
  },
  'sentence_486': {
    'answer': 'Steven bought nine green desks',
    'testVideos':
        ['Structured Sentences/F1/Steven bought nine green desks_1.mp4'],
  },
  'sentence_487': {
    'answer': 'William got three pretty desks',
    'testVideos':
        ['Structured Sentences/F1/William got three pretty desks_1.mp4'],
  },
  'sentence_488': {
    'answer': 'Thomas wants seven green toys',
    'testVideos':
        ['Structured Sentences/F1/Thomas wants seven green toys_1.mp4'],
  },
  'sentence_489': {
    'answer': 'Allen got three green chairs',
    'testVideos':
        ['Structured Sentences/F1/Allen got three green chairs_1.mp4'],
  },
  'sentence_490': {
    'answer': 'Rachel sold three red windows',
    'testVideos':
        ['Structured Sentences/F1/Rachel sold three red windows_1.mp4'],
  },
  'sentence_491': {
    'answer': 'Allen wants fifteen large houses',
    'testVideos':
        ['Structured Sentences/F1/Allen wants fifteen large houses_1.mp4'],
  },
  'sentence_492': {
    'answer': 'Peter bought three cheap rings',
    'testVideos':
        ['Structured Sentences/F1/Peter bought three cheap rings_1.mp4'],
  },
  'sentence_493': {
    'answer': 'Allen wants nineteen white desks',
    'testVideos':
        ['Structured Sentences/F1/Allen wants nineteen white desks_1.mp4'],
  },
  'sentence_494': {
    'answer': 'Peter got three small windows',
    'testVideos':
        ['Structured Sentences/F1/Peter got three small windows_1.mp4'],
  },
  'sentence_495': {
    'answer': 'Peter got eight white windows',
    'testVideos':
        ['Structured Sentences/F1/Peter got eight white windows_1.mp4'],
  },
  'sentence_496': {
    'answer': 'Peter ordered four red toys',
    'testVideos': ['Structured Sentences/F1/Peter ordered four red toys_1.mp4'],
  },
  'sentence_497': {
    'answer': 'Peter sees seven large desks',
    'testVideos':
        ['Structured Sentences/F1/Peter sees seven large desks_1.mp4'],
  },
  'sentence_498': {
    'answer': 'Peter kept three white desks',
    'testVideos':
        ['Structured Sentences/F1/Peter kept three white desks_1.mp4'],
  },
  'sentence_499': {
    'answer': 'Rachel got three heavy sofas',
    'testVideos':
        ['Structured Sentences/F1/Rachel got three heavy sofas_1.mp4'],
  },
  'sentence_500': {
    'answer': 'Kathy bought three small toys',
    'testVideos':
        ['Structured Sentences/F1/Kathy bought three small toys_1.mp4'],
  },
  'cuny_1_1': {
    'answer': 'Have you eaten yet?',
    'testVideos': ['CUNY/cuny1/cuny1-1proc_converted2.mov'],
  },
  'cuny_1_2': {
    'answer': 'How many of your brothers still live at home?',
    'testVideos': ['CUNY/cuny1/cuny1-2proc_converted2.mov'],
  },
  'cuny_1_3': {
    'answer': 'How many years of school did it take to become a nurse?',
    'testVideos': ['CUNY/cuny1/cuny1-3proc_converted2.mov'],
  },
  'cuny_1_4': {
    'answer': 'Where can I get my suit cleaned?',
    'testVideos': ['CUNY/cuny1/cuny1-4proc_converted2.mov'],
  },
  'cuny_1_5': {
    'answer':
        'Cats are easy to take care of because you don\'t have to walk them.',
    'testVideos': ['CUNY/cuny1/cuny1-5proc_converted2.mov'],
  },
  'cuny_1_6': {
    'answer': 'She just moved into a three room apartment.',
    'testVideos': ['CUNY/cuny1/cuny1-6proc_converted2.mov'],
  },
  'cuny_1_7': {
    'answer': 'I like to play tennis.',
    'testVideos': ['CUNY/cuny1/cuny1-7proc_converted2.mov'],
  },
  'cuny_1_8': {
    'answer': 'We couldn\'t fly home yesterday because of the big snowstorm.',
    'testVideos': ['CUNY/cuny1/cuny1-8proc_converted2.mov'],
  },
  'cuny_1_9': {
    'answer': 'Remember to get plenty of rest and drink lots of fluids.',
    'testVideos': ['CUNY/cuny1/cuny1-9proc_converted2.mov'],
  },
  'cuny_1_10': {
    'answer': 'Carve the turkey.',
    'testVideos': ['CUNY/cuny1/cuny1-10proc_converted2.mov'],
  },
  'cuny_1_11': {
    'answer': 'Make sure you deposit that check.',
    'testVideos': ['CUNY/cuny1/cuny1-11proc_converted2.mov'],
  },
  'cuny_1_12': {
    'answer':
        'Please make sure that you practice a lot before your next piano lesson.',
    'testVideos': ['CUNY/cuny1/cuny1-12proc_converted2.mov'],
  },
  'cuny_2_1': {
    'answer': 'Do you want to have a barbecue this evening?',
    'testVideos': ['CUNY/cuny2/cuny2-1proc_converted2.mov'],
  },
  'cuny_2_2': {
    'answer': 'When was the last time that you went to visit your parents?',
    'testVideos': ['CUNY/cuny2/cuny2-2proc_converted2.mov'],
  },
  'cuny_2_3': {
    'answer': 'When will you be taking your vacation?',
    'testVideos': ['CUNY/cuny2/cuny2-3proc_converted2.mov'],
  },
  'cuny_2_4': {
    'answer':
        'Remember to take enough hangers with you when you go to do your laundry.',
    'testVideos': ['CUNY/cuny2/cuny2-4proc_converted2.mov'],
  },
  'cuny_2_5': {
    'answer': 'You can see deer near my country house.',
    'testVideos': ['CUNY/cuny2/cuny2-5proc_converted2.mov'],
  },
  'cuny_2_6': {
    'answer': 'We\'re looking for an apartment.',
    'testVideos': ['CUNY/cuny2/cuny2-6proc_converted2.mov'],
  },
  'cuny_2_7': {
    'answer': 'The football field is right next to the baseball field.',
    'testVideos': ['CUNY/cuny2/cuny2-7proc_converted2.mov'],
  },
  'cuny_2_8': {
    'answer': 'Can you remember the last time we had so much snow?',
    'testVideos': ['CUNY/cuny2/cuny2-8proc_converted2.mov'],
  },
  'cuny_2_9': {
    'answer': 'See your doctor.',
    'testVideos': ['CUNY/cuny2/cuny2-9proc_converted2.mov'],
  },
  'cuny_2_10': {
    'answer': 'Put these lights on the tree.',
    'testVideos': ['CUNY/cuny2/cuny2-10proc_converted2.mov'],
  },
  'cuny_2_11': {
    'answer': 'Don\'t use your credit card if you can\'t pay the bill on time.',
    'testVideos': ['CUNY/cuny2/cuny2-11proc_converted2.mov'],
  },
  'cuny_2_12': {
    'answer': 'I like that song.',
    'testVideos': ['CUNY/cuny2/cuny2-12proc_converted2.mov'],
  },
  'cuny_3_1': {
    'answer': 'Do you want fried chicken or do you want pizza for dinner?',
    'testVideos': ['CUNY/cuny3/cuny3-1proc_converted2.mov'],
  },
  'cuny_3_2': {
    'answer': 'How long have your parents been married?',
    'testVideos': ['CUNY/cuny3/cuny3-2proc_converted2.mov'],
  },
  'cuny_3_3': {
    'answer':
        'Take the job only if you feel that the work will be more challenging.',
    'testVideos': ['CUNY/cuny3/cuny3-3proc_converted2.mov'],
  },
  'cuny_3_4': {
    'answer': 'Buy that bathing suit because it fits you.',
    'testVideos': ['CUNY/cuny3/cuny3-4proc_converted2.mov'],
  },
  'cuny_3_5': {
    'answer': 'That dog likes to run.',
    'testVideos': ['CUNY/cuny3/cuny3-5proc_converted2.mov'],
  },
  'cuny_3_6': {
    'answer': 'We\'re going to paint the guest bedroom next week.',
    'testVideos': ['CUNY/cuny3/cuny3-6proc_converted2.mov'],
  },
  'cuny_3_7': {
    'answer': 'How many miles a day do you run before a marathon.',
    'testVideos': ['CUNY/cuny3/cuny3-7proc_converted2.mov'],
  },
  'cuny_3_8': {
    'answer': 'Did it rain?',
    'testVideos': ['CUNY/cuny3/cuny3-8proc_converted2.mov'],
  },
  'cuny_3_9': {
    'answer': 'Remember to take all your vitamins.',
    'testVideos': ['CUNY/cuny3/cuny3-9proc_converted2.mov'],
  },
  'cuny_3_10': {
    'answer': 'You buy the food and she will buy the drinks for the barbecue.',
    'testVideos': ['CUNY/cuny3/cuny3-10proc_converted2.mov'],
  },
  'cuny_3_11': {
    'answer': 'That place is expensive.',
    'testVideos': ['CUNY/cuny3/cuny3-11proc_converted2.mov'],
  },
  'cuny_3_12': {
    'answer': 'The rock concerts on the pier have been very successful.',
    'testVideos': ['CUNY/cuny3/cuny3-12proc_converted2.mov'],
  },
  'cuny_4_1': {
    'answer': 'Do you want gravy on your potatoes?',
    'testVideos': ['CUNY/cuny4/cuny4-1proc_converted2.mov'],
  },
  'cuny_4_2': {
    'answer':
        'You have to invite all of your aunts and uncles to the wedding reception.',
    'testVideos': ['CUNY/cuny4/cuny4-2proc_converted2.mov'],
  },
  'cuny_4_3': {
    'answer': 'Make sure you get to work on time.',
    'testVideos': ['CUNY/cuny4/cuny4-3proc_converted2.mov'],
  },
  'cuny_4_4': {
    'answer': 'Throw away your old shoes.',
    'testVideos': ['CUNY/cuny4/cuny4-4proc_converted2.mov'],
  },
  'cuny_4_5': {
    'answer': 'My cat was chasing the bird all around the yard.',
    'testVideos': ['CUNY/cuny4/cuny4-5proc_converted2.mov'],
  },
  'cuny_4_6': {
    'answer': 'Did it take you a long time to find an apartment?',
    'testVideos': ['CUNY/cuny4/cuny4-6proc_converted2.mov'],
  },
  'cuny_4_7': {
    'answer': 'Do you jog?',
    'testVideos': ['CUNY/cuny4/cuny4-7proc_converted2.mov'],
  },
  'cuny_4_8': {
    'answer': 'Did you bring an umbrella today?',
    'testVideos': ['CUNY/cuny4/cuny4-8proc_converted2.mov'],
  },
  'cuny_4_9': {
    'answer':
        'Don\'t drink alcohol while you are pregnant because it\'s harmful to your baby.',
    'testVideos': ['CUNY/cuny4/cuny4-9proc_converted2.mov'],
  },
  'cuny_4_10': {
    'answer': 'Summer is finally here.',
    'testVideos': ['CUNY/cuny4/cuny4-10proc_converted2.mov'],
  },
  'cuny_4_11': {
    'answer': 'I lost my checkbook so I closed the account.',
    'testVideos': ['CUNY/cuny4/cuny4-11proc_converted2.mov'],
  },
  'cuny_4_12': {
    'answer': 'He did not learn how to play the piano until very recently.',
    'testVideos': ['CUNY/cuny4/cuny4-12proc_converted2.mov'],
  },
  'cuny_5_1': {
    'answer':
        'Please get the groceries out of the car and put them away for me.',
    'testVideos': ['CUNY/cuny5/cuny5-1proc_converted2.mov'],
  },
  'cuny_5_2': {
    'answer': 'Make sure you call your brother this week.',
    'testVideos': ['CUNY/cuny5/cuny5-2proc_converted2.mov'],
  },
  'cuny_5_3': {
    'answer': 'Finish that report this afternoon.',
    'testVideos': ['CUNY/cuny5/cuny5-3proc_converted2.mov'],
  },
  'cuny_5_4': {
    'answer': 'Take the dress back and buy something that you like.',
    'testVideos': ['CUNY/cuny5/cuny5-4proc_converted2.mov'],
  },
  'cuny_5_5': {
    'answer': 'Did your landlord give you any trouble about having a dog.',
    'testVideos': ['CUNY/cuny5/cuny5-5proc_converted2.mov'],
  },
  'cuny_5_6': {
    'answer': 'Where\'s the bathroom?',
    'testVideos': ['CUNY/cuny5/cuny5-6proc_converted2.mov'],
  },
  'cuny_5_7': {
    'answer': 'Do you like to go hiking?',
    'testVideos': ['CUNY/cuny5/cuny5-7proc_converted2.mov'],
  },
  'cuny_5_8': {
    'answer': 'Do you know if it rains a lot at this time of year?',
    'testVideos': ['CUNY/cuny5/cuny5-8proc_converted2.mov'],
  },
  'cuny_5_9': {
    'answer': 'She has a fever.',
    'testVideos': ['CUNY/cuny5/cuny5-9proc_converted2.mov'],
  },
  'cuny_5_10': {
    'answer': 'I like going to the mountains in the fall.',
    'testVideos': ['CUNY/cuny5/cuny5-10proc_converted2.mov'],
  },
  'cuny_5_11': {
    'answer': 'Many of my friends had to take out school loans this year.',
    'testVideos': ['CUNY/cuny5/cuny5-11proc_converted2.mov'],
  },
  'cuny_5_12': {
    'answer': 'Their last song was a big hit.',
    'testVideos': ['CUNY/cuny5/cuny5-12proc_converted2.mov'],
  },
  'cuny_6_1': {
    'answer': 'Make sure that you don\'t overcook the shrimp.',
    'testVideos': ['CUNY/cuny6/cuny6-1proc_converted2.mov'],
  },
  'cuny_6_2': {
    'answer': 'Visit your grandmother on Sunday.',
    'testVideos': ['CUNY/cuny6/cuny6-2proc_converted2.mov'],
  },
  'cuny_6_3': {
    'answer': 'If you want to quit, please give two week’s notice.',
    'testVideos': ['CUNY/cuny6/cuny6-3proc_converted2.mov'],
  },
  'cuny_6_4': {
    'answer': 'The store is having a sale on all their bathing suits.',
    'testVideos': ['CUNY/cuny6/cuny6-4proc_converted2.mov'],
  },
  'cuny_6_5': {
    'answer': 'Where\'s your dog?',
    'testVideos': ['CUNY/cuny6/cuny6-5proc_converted2.mov'],
  },
  'cuny_6_6': {
    'answer': 'How many bedrooms do you have?',
    'testVideos': ['CUNY/cuny6/cuny6-6proc_converted2.mov'],
  },
  'cuny_6_7': {
    'answer':
        'Did you ever think about trying to sell some of your better paintings?',
    'testVideos': ['CUNY/cuny6/cuny6-7proc_converted2.mov'],
  },
  'cuny_6_8': {
    'answer': 'Put on your raincoat.',
    'testVideos': ['CUNY/cuny6/cuny6-8proc_converted2.mov'],
  },
  'cuny_6_9': {
    'answer': 'All my friends seem to be catching summer colds.',
    'testVideos': ['CUNY/cuny6/cuny6-9proc_converted2.mov'],
  },
  'cuny_6_10': {
    'answer': 'It seems like we haven\'t had a decent spring in several years.',
    'testVideos': ['CUNY/cuny6/cuny6-10proc_converted2.mov'],
  },
  'cuny_6_11': {
    'answer': 'She wants to start buying some stock.',
    'testVideos': ['CUNY/cuny6/cuny6-11proc_converted2.mov'],
  },
  'cuny_6_12': {
    'answer':
        'Did you know that they both get season tickets for the opera every year?',
    'testVideos': ['CUNY/cuny6/cuny6-12proc_converted2.mov'],
  },
  'cuny_7_1': {
    'answer': 'Please turn on the coffee.',
    'testVideos': ['CUNY/cuny7/cuny7-1proc_converted2.mov'],
  },
  'cuny_7_2': {
    'answer': 'You have to let your children make their own mistakes.',
    'testVideos': ['CUNY/cuny7/cuny7-2proc_converted2.mov'],
  },
  'cuny_7_3': {
    'answer': 'If you work for an airline you get great travel benefits.',
    'testVideos': ['CUNY/cuny7/cuny7-3proc_converted2.mov'],
  },
  'cuny_7_4': {
    'answer': 'My zipper broke.',
    'testVideos': ['CUNY/cuny7/cuny7-4proc_converted2.mov'],
  },
  'cuny_7_5': {
    'answer': 'How many cats do you have?',
    'testVideos': ['CUNY/cuny7/cuny7-5proc_converted2.mov'],
  },
  'cuny_7_6': {
    'answer':
        'Would you ever consider building an extra room or two onto the house?',
    'testVideos': ['CUNY/cuny7/cuny7-6proc_converted2.mov'],
  },
  'cuny_7_7': {
    'answer': 'Throw me the ball.',
    'testVideos': ['CUNY/cuny7/cuny7-7proc_converted2.mov'],
  },
  'cuny_7_8': {
    'answer': 'Be careful driving on the bridge when it\'s windy.',
    'testVideos': ['CUNY/cuny7/cuny7-8proc_converted2.mov'],
  },
  'cuny_7_9': {
    'answer': 'Children get sick more often when they start to go to school.',
    'testVideos': ['CUNY/cuny7/cuny7-9proc_converted2.mov'],
  },
  'cuny_7_10': {
    'answer': 'I always eat a lot on Thanksgiving.',
    'testVideos': ['CUNY/cuny7/cuny7-10proc_converted2.mov'],
  },
  'cuny_7_11': {
    'answer':
        'What type of investment plan did they think would be the best for you?',
    'testVideos': ['CUNY/cuny7/cuny7-11proc_converted2.mov'],
  },
  'cuny_7_12': {
    'answer': 'Have you seen the new musical on Broadway?',
    'testVideos': ['CUNY/cuny7/cuny7-12proc_converted2.mov'],
  },
  'cuny_8_1': {
    'answer': 'Make sure you wash the fruit before you eat it.',
    'testVideos': ['CUNY/cuny8/cuny8-1proc_converted2.mov'],
  },
  'cuny_8_2': {
    'answer': 'My brother and his two children are coming to visit me.',
    'testVideos': ['CUNY/cuny8/cuny8-2proc_converted2.mov'],
  },
  'cuny_8_3': {
    'answer': 'Here\'s my office.',
    'testVideos': ['CUNY/cuny8/cuny8-3proc_converted2.mov'],
  },
  'cuny_8_4': {
    'answer': 'I need to buy a suit.',
    'testVideos': ['CUNY/cuny8/cuny8-4proc_converted2.mov'],
  },
  'cuny_8_5': {
    'answer':
        'Did you see any interesting birds when you went bird watching last week?',
    'testVideos': ['CUNY/cuny8/cuny8-5proc_converted2.mov'],
  },
  'cuny_8_6': {
    'answer': 'Clean the kitchen first.',
    'testVideos': ['CUNY/cuny8/cuny8-6proc_converted2.mov'],
  },
  'cuny_8_7': {
    'answer': 'Don\'t try to run unless you have good shoes.',
    'testVideos': ['CUNY/cuny8/cuny8-7proc_converted2.mov'],
  },
  'cuny_8_8': {
    'answer':
        'Always take your sunglasses with you when you drive on sunny days.',
    'testVideos': ['CUNY/cuny8/cuny8-8proc_converted2.mov'],
  },
  'cuny_8_9': {
    'answer': 'You have to start losing some weight.',
    'testVideos': ['CUNY/cuny8/cuny8-9proc_converted2.mov'],
  },
  'cuny_8_10': {
    'answer':
        'Do you want to take your vacation in the winter or summer this year?',
    'testVideos': ['CUNY/cuny8/cuny8-10proc_converted2.mov'],
  },
  'cuny_8_11': {
    'answer': 'Do you always balance your checkbook every month?',
    'testVideos': ['CUNY/cuny8/cuny8-11proc_converted2.mov'],
  },
  'cuny_8_12': {
    'answer': 'Do you play the guitar?',
    'testVideos': ['CUNY/cuny8/cuny8-12proc_converted2.mov'],
  },
  'cuny_9_1': {
    'answer':
        'The desserts at that restaurant are very good but very expensive.',
    'testVideos': ['CUNY/cuny9/cuny9-1proc_converted2.mov'],
  },
  'cuny_9_2': {
    'answer': 'She\'s my daughter.',
    'testVideos': ['CUNY/cuny9/cuny9-2proc_converted2.mov'],
  },
  'cuny_9_3': {
    'answer': 'He doesn\'t like his new boss.',
    'testVideos': ['CUNY/cuny9/cuny9-3proc_converted2.mov'],
  },
  'cuny_9_4': {
    'answer': 'I bought three pairs of shoes and a pair of boots last week.',
    'testVideos': ['CUNY/cuny9/cuny9-4proc_converted2.mov'],
  },
  'cuny_9_5': {
    'answer': 'Let the dog out.',
    'testVideos': ['CUNY/cuny9/cuny9-5proc_converted2.mov'],
  },
  'cuny_9_6': {
    'answer': 'Move the couch to the wall facing the window.',
    'testVideos': ['CUNY/cuny9/cuny9-6proc_converted2.mov'],
  },
  'cuny_9_7': {
    'answer':
        'If you want to take good pictures, take a few photography classes.',
    'testVideos': ['CUNY/cuny9/cuny9-7proc_converted2.mov'],
  },
  'cuny_9_8': {
    'answer': 'Wear a heavy sweater since it\'s cold.',
    'testVideos': ['CUNY/cuny9/cuny9-8proc_converted2.mov'],
  },
  'cuny_9_9': {
    'answer': 'Do you go to a gym to exercise, or do you exercise at home?',
    'testVideos': ['CUNY/cuny9/cuny9-9proc_converted2.mov'],
  },
  'cuny_9_10': {
    'answer': 'Have you finished all your Christmas shopping yet?',
    'testVideos': ['CUNY/cuny9/cuny9-10proc_converted2.mov'],
  },
  'cuny_9_11': {
    'answer': 'Can I pay by check?',
    'testVideos': ['CUNY/cuny9/cuny9-11proc_converted2.mov'],
  },
  'cuny_9_12': {
    'answer': 'Do you think he\'ll play the piano at the party?',
    'testVideos': ['CUNY/cuny9/cuny9-12proc_converted2.mov'],
  },
  'cuny_10_1': {
    'answer': 'Cake is sweet.',
    'testVideos': ['CUNY/cuny10/cuny10-1proc_converted2.mov'],
  },
  'cuny_10_2': {
    'answer': 'My sister has a new boyfriend.',
    'testVideos': ['CUNY/cuny10/cuny10-2proc_converted2.mov'],
  },
  'cuny_10_3': {
    'answer': 'He got a promotion at work and now he doesn\'t want to leave.',
    'testVideos': ['CUNY/cuny10/cuny10-3proc_converted2.mov'],
  },
  'cuny_10_4': {
    'answer': 'Is that coat new?',
    'testVideos': ['CUNY/cuny10/cuny10-4proc_converted2.mov'],
  },
  'cuny_10_5': {
    'answer': 'Please don\'t go that close to the lion\'s cage.',
    'testVideos': ['CUNY/cuny10/cuny10-5proc_converted2.mov'],
  },
  'cuny_10_6': {
    'answer': 'Remember to lock all the doors and windows before you go away.',
    'testVideos': ['CUNY/cuny10/cuny10-6proc_converted2.mov'],
  },
  'cuny_10_7': {
    'answer': 'Take your baseball glove to the game.',
    'testVideos': ['CUNY/cuny10/cuny10-7proc_converted2.mov'],
  },
  'cuny_10_8': {
    'answer':
        'All the roads into the city were closed yesterday due to the heavy rains.',
    'testVideos': ['CUNY/cuny10/cuny10-8proc_converted2.mov'],
  },
  'cuny_10_9': {
    'answer': 'I know a doctor who makes house calls.',
    'testVideos': ['CUNY/cuny10/cuny10-9proc_converted2.mov'],
  },
  'cuny_10_10': {
    'answer': 'Where are you spending Thanksgiving?',
    'testVideos': ['CUNY/cuny10/cuny10-10proc_converted2.mov'],
  },
  'cuny_10_11': {
    'answer': 'How much interest are they charging you on your loan?',
    'testVideos': ['CUNY/cuny10/cuny10-11proc_converted2.mov'],
  },
  'cuny_10_12': {
    'answer': 'Don\'t stand so close to the microphone when you are singing.',
    'testVideos': ['CUNY/cuny10/cuny10-12proc_converted2.mov'],
  },
  'cuny_11_1': {
    'answer': 'She baked a big apple pie.',
    'testVideos': ['CUNY/cuny11/cuny11-1proc_converted2.mov'],
  },
  'cuny_11_2': {
    'answer':
        'She shares a two bedroom apartment in the city with her two sisters.',
    'testVideos': ['CUNY/cuny11/cuny11-2proc_converted2.mov'],
  },
  'cuny_11_3': {
    'answer': 'How\'s your new job?',
    'testVideos': ['CUNY/cuny11/cuny11-3proc_converted2.mov'],
  },
  'cuny_11_4': {
    'answer': 'How much did they charge you for the alterations?',
    'testVideos': ['CUNY/cuny11/cuny11-4proc_converted2.mov'],
  },
  'cuny_11_5': {
    'answer': 'Take the dog to the vet every year for its rabies shot.',
    'testVideos': ['CUNY/cuny11/cuny11-5proc_converted2.mov'],
  },
  'cuny_11_6': {
    'answer': 'Clean the guest bedroom before next weekend.',
    'testVideos': ['CUNY/cuny11/cuny11-6proc_converted2.mov'],
  },
  'cuny_11_7': {
    'answer':
        'Professional football players usually have to train in the summer before the season begins.',
    'testVideos': ['CUNY/cuny11/cuny11-7proc_converted2.mov'],
  },
  'cuny_11_8': {
    'answer': 'When the humidity is high, it\'s uncomfortable outside.',
    'testVideos': ['CUNY/cuny11/cuny11-8proc_converted2.mov'],
  },
  'cuny_11_9': {
    'answer': 'What medicine are you taking?',
    'testVideos': ['CUNY/cuny11/cuny11-9proc_converted2.mov'],
  },
  'cuny_11_10': {
    'answer': 'Do you want to go Christmas shopping with me today?',
    'testVideos': ['CUNY/cuny11/cuny11-10proc_converted2.mov'],
  },
  'cuny_11_11': {
    'answer': 'You really should be able to save more of your salary.',
    'testVideos': ['CUNY/cuny11/cuny11-11proc_converted2.mov'],
  },
  'cuny_11_12': {
    'answer': 'Play that song.',
    'testVideos': ['CUNY/cuny11/cuny11-12proc_converted2.mov'],
  },
  'cuny_12_1': {
    'answer':
        'We had roast beef, baked potatoes with sour cream and broccoli for dinner.',
    'testVideos': ['CUNY/cuny12/cuny12-1proc_converted2.mov'],
  },
  'cuny_12_2': {
    'answer': 'How\'s your brother doing?',
    'testVideos': ['CUNY/cuny12/cuny12-2proc_converted2.mov'],
  },
  'cuny_12_3': {
    'answer': 'How long have you been working for this company?',
    'testVideos': ['CUNY/cuny12/cuny12-3proc_converted2.mov'],
  },
  'cuny_12_4': {
    'answer': 'Did you know that she made her own dress for the wedding?',
    'testVideos': ['CUNY/cuny12/cuny12-4proc_converted2.mov'],
  },
  'cuny_12_5': {
    'answer': 'Don\'t let the dog off his leash.',
    'testVideos': ['CUNY/cuny12/cuny12-5proc_converted2.mov'],
  },
  'cuny_12_6': {
    'answer':
        'We decided to paint the living room light blue and the dining room yellow.',
    'testVideos': ['CUNY/cuny12/cuny12-6proc_converted2.mov'],
  },
  'cuny_12_7': {
    'answer': 'He plays softball with his friends on Sundays.',
    'testVideos': ['CUNY/cuny12/cuny12-7proc_converted2.mov'],
  },
  'cuny_12_8': {
    'answer': 'The sun is finally shining.',
    'testVideos': ['CUNY/cuny12/cuny12-8proc_converted2.mov'],
  },
  'cuny_12_9': {
    'answer': 'Did he get sick from eating some bad food yesterday?',
    'testVideos': ['CUNY/cuny12/cuny12-9proc_converted2.mov'],
  },
  'cuny_12_10': {
    'answer': 'Be careful if you use fireworks on the Fourth of July.',
    'testVideos': ['CUNY/cuny12/cuny12-10proc_converted2.mov'],
  },
  'cuny_12_11': {
    'answer': 'Take your change.',
    'testVideos': ['CUNY/cuny12/cuny12-11proc_converted2.mov'],
  },
  'cuny_12_12': {
    'answer': 'Put your albums where they belong.',
    'testVideos': ['CUNY/cuny12/cuny12-12proc_converted2.mov'],
  },
};
