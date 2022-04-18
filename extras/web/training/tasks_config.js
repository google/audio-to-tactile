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
 * This file provides two configuration constants:
 *
 * taskTypes defines the required states for UI controls for common task types.
 * Within each task type, each UI control is provided with a default value and
 * whether or not it is allowed to be changed by the user.
 * - Parameter names must match exactly the class name of the UI element.
 * - The first entry for each parameter defines the default value.
 *   Default values can be:
 *      - booleans (e.g. state of a checkbox)
 *      - numbers (e.g. value of a number entry field)
 *      - null (indicating the element has no value, e.g. buttons)
 * - The second entry defines whether the user can change the value. This is
 *   always a boolean; 'false' will disable user control.
 *
 * Protcols define a re-usable ordering of task types, with:
 * - the number of presentations for the taks,
 * - a title,
 * - a video group (test, pretest, or train)
 * - whether to mute or play video audio.
 *     In an L condition, audio should be muted; in an  L+T condition, audio
 *     should play for the device.
 *
 * tasks contains specific experiment definitions.  Each entry defines:
 * - a session name
 * - a list of tasks via a protocol entry
 * - a list of pairs of stimuli to perform each protocol for. Pairs contain:
 *    - a name for the pair
 *    - the stimuli
 *    - correct answers. This is a list of lists of options that should be
 *       included in a multiple choice question for this word. This always
 *       includes the correct answer. If multiple lists are provided, each list
 *       represents a column for a grid choice, where one answer must be
 * selected from each column.
 */

const taskTypes = {
  test: {
    'train-button': [null, false],
    'practice-button': [null, false],
    'test-button': [null, true],
    'answer-mode': [true, false],
    'shuffle-flag': [true, false],
    'num-repeats': [1, false],
    'without-caption': [false, false],
    'display-results': [false, false],
    'allow-replay': [false, false],
    'word-selector': [null, false],
  },
  freePlay: {
    'train-button': [null, true],
    'practice-button': [null, false],
    'test-button': [null, false],
    'answer-mode': [true, false],
    'shuffle-flag': [true, false],
    'num-repeats': [1, false],
    'without-caption': [false, false],
    'display-results': [false, false],
    'allow-replay': [false, false],
    'word-selector': [null, true],
  },
  feedback: {
    'train-button': [null, false],
    'practice-button': [null, true],
    'test-button': [null, false],
    'answer-mode': [true, false],
    'shuffle-flag': [true, false],
    'num-repeats': [1, false],
    'without-caption': [false, false],
    'display-results': [false, false],
    'allow-replay': [true, false],
    'word-selector': [null, false],
  },
  userControl: {
    'train-button': [null, true],
    'practice-button': [null, true],
    'test-button': [null, true],
    'answer-mode': [true, true],
    'shuffle-flag': [true, true],
    'num-repeats': [1, true],
    'without-caption': [false, true],
    'display-results': [false, true],
    'allow-replay': [false, true],
    'word-selector': [null, true],
  },
  freeResponseFeedback: {
    'train-button': [null, false],
    'practice-button': [null, true],
    'test-button': [null, false],
    'answer-mode': [false, false],
    'shuffle-flag': [true, false],
    'num-repeats': [1, false],
    'without-caption': [false, false],
    'display-results': [false, false],
    'allow-replay': [true, false],
    'word-selector': [null, false],
  },
  freeResponseTest: {
    'train-button': [null, false],
    'practice-button': [null, false],
    'test-button': [null, true],
    'answer-mode': [false, false],
    'shuffle-flag': [true, false],
    'num-repeats': [1, false],
    'without-caption': [false, false],
    'display-results': [false, false],
    'allow-replay': [false, false],
    'word-selector': [null, false],
  },


};

const discriminationProtocol = [
  {
    taskType: 'test',
    withReplacement: true,
    numPresentations: 20,
    title: 'Pre-Test (L)',
    videoGroup: 'preTestVideos',
    mute: true,
  },
  {
    taskType: 'freePlay',
    withReplacement: true,
    numPresentations: 20,
    title: 'Free Play (L+T)',
    videoGroup: 'trainVideos',
    mute: false,
  },
  {
    taskType: 'feedback',
    withReplacement: true,
    numPresentations: 20,
    title: 'Feedback (L)',
    videoGroup: 'trainVideos',
    mute: true,
  },
  {
    taskType: 'feedback',
    withReplacement: true,
    numPresentations: 20,
    title: 'Feedback (L+T)',
    videoGroup: 'trainVideos',
    mute: false,
  },
  {
    taskType: 'test',
    withReplacement: true,
    numPresentations: 40,
    title: 'Post-Test (L+T)',
    videoGroup: 'testVideos',
    mute: false,
  },
  {
    taskType: 'test',
    withReplacement: true,
    numPresentations: 40,
    title: 'Post-Test (L)',
    videoGroup: 'testVideos',
    mute: true,
  },
];

const identificationProtocol = [
  {
    taskType: 'feedback',
    withReplacement: true,
    numPresentations: 20,
    title: 'Feedback (L)',
    videoGroup: 'trainVideos',
    mute: true,
  },
  {
    taskType: 'feedback',
    withReplacement: true,
    numPresentations: 20,
    title: 'Feedback (L+T)',
    videoGroup: 'trainVideos',
    mute: false,
  },
  {
    taskType: 'test',
    withReplacement: true,
    numPresentations: 20,
    title: 'Post-Test (L+T)',
    videoGroup: 'testVideos',
    mute: false,
  },
  {
    taskType: 'test',
    withReplacement: true,
    numPresentations: 20,
    title: 'Post-Test (L)',
    videoGroup: 'testVideos',
    mute: true,
  },
];

const structuredSentencesTrainLProtocol = [{
  taskType: 'feedback',
  withReplacement: false,
  numPresentations: 10,
  title: 'Feedback (L)',
  videoGroup: 'testVideos',
  mute: true,
}];

const structuredSentencesTrainLTProtocol = [{
  taskType: 'feedback',
  withReplacement: false,
  numPresentations: 10,
  title: 'Feedback (L + T)',
  videoGroup: 'testVideos',
  mute: false,
}];


const structuredSentencesTestLProtocol = [{
  taskType: 'test',
  withReplacement: false,
  numPresentations: 10,
  title: 'Post-Test (L)',
  videoGroup: 'testVideos',
  mute: true,
}];

const structuredSentencesTestLTProtocol = [{
  taskType: 'test',
  withReplacement: false,
  numPresentations: 10,
  title: 'Post-Test (L+T)',
  videoGroup: 'testVideos',
  mute: false,
}];

const freeSentencesTrainLProtocol = [{
  taskType: 'freeResponseFeedback',
  withReplacement: false,
  numPresentations: 12,
  title: 'Feedback (L)',
  videoGroup: 'testVideos',
  mute: true,
}];

const freeSentencesTrainLTProtocol = [{
  taskType: 'freeResponseFeedback',
  withReplacement: false,
  numPresentations: 12,
  title: 'Feedback (L + T)',
  videoGroup: 'testVideos',
  mute: false,
}];


const freeSentencesTestLProtocol = [{
  taskType: 'freeResponseTest',
  withReplacement: false,
  numPresentations: 12,
  title: 'Post-Test (L)',
  videoGroup: 'testVideos',
  mute: true,
}];

const freeSentencesTestLTProtocol = [{
  taskType: 'freeResponseTest',
  withReplacement: false,
  numPresentations: 12,
  title: 'Post-Test (L+T)',
  videoGroup: 'testVideos',
  mute: false,
}];



const tasks = [
  {
    sessionName: 'Discrimination - Group 1',
    tasks: discriminationProtocol,
    pairs: [
      {
        title: 'p/b',
        stimuli: [
          'bah', 'bay', 'bee', 'boh', 'boo', 'pah', 'pay', 'pee', 'poh', 'poo'
        ],
        answers: [['b', 'p']],
      },
      {
        title: 'p/m',
        stimuli: [
          'mah', 'may', 'mee', 'moh', 'moo', 'pah', 'pay', 'pee', 'poh', 'poo'
        ],
        answers: [['p', 'm']],
      },
      {
        title: 'b/m',
        stimuli: [
          'mah', 'may', 'mee', 'moh', 'moo', 'bah', 'bay', 'bee', 'boh', 'boo'
        ],
        answers: [['m', 'b']],
      },
      {
        title: 'f/v',
        stimuli: [
          'fah', 'fay', 'fee', 'foh', 'foo', 'vah', 'vay', 'vee', 'voh', 'voo'
        ],
        answers: [['f', 'v']],
      },
      {
        title: 'θ/ð',
        stimuli: [
          'thah', 'thay', 'thee', 'thoh', 'thoo', 'dhah', 'dhay', 'dhee',
          'dhoh', 'dhoo'
        ],
        answers: [['th', 'dh']],
      },
    ],
  },
  {
    sessionName: 'Discrimination - Group 2',
    tasks: discriminationProtocol,
    pairs: [
      {
        title: 's/z',
        stimuli: [
          'sah', 'say', 'see', 'soh', 'soo', 'zah', 'zay', 'zee', 'zoh', 'zoo'
        ],
        answers: [['s', 'z']],
      },
      {
        title: '∫ / ʒ',
        stimuli: [
          'shah', 'shay', 'shee', 'shoh', 'shoo', 'zhah', 'zhay', 'zhee',
          'zhoh', 'zhoo'
        ],
        answers: [['sh', 'zh']],
      },
      {
        title: 'tʃ / ʤ',
        stimuli: [
          'chah', 'chay', 'chee', 'choh', 'choo', 'jah', 'jay', 'jee', 'joh',
          'joo'
        ],
        answers: [['', '']],
      },
      {
        title: 'w/r',
        stimuli: [
          'wah', 'way', 'wee', 'woh', 'woo', 'rah', 'ray', 'ree', 'roh', 'roo'
        ],
        answers: [['w', 'r']],
      },
      {
        title: 'l/n',
        stimuli: [
          'lah', 'lay', 'lee', 'loh', 'loo', 'nah', 'nay', 'nee', 'noh', 'noo'
        ],
        answers: [['l', 'n']],
      },
    ],
  },
  {
    sessionName: 'Discrimination - Group 3',
    tasks: discriminationProtocol,
    pairs: [
      {
        title: 't/d',
        stimuli: [
          'tah', 'tay', 'tee', 'toh', 'too', 'dah', 'day', 'dee', 'doh', 'doo'
        ],
        answers: [['t', 'd']],
      },
      {
        title: 't/n',
        stimuli: [
          'tah', 'tay', 'tee', 'toh', 'too', 'nah', 'nay', 'nee', 'noh', 'noo'
        ],
        answers: [['t', 'n']],
      },
      {
        title: 't/k',
        stimuli: [
          'tah', 'tay', 'tee', 'toh', 'too', 'dah', 'day', 'dee', 'doh', 'doo'
        ],
        answers: [['t', 'k']],
      },
      {
        title: 't/g',
        stimuli: [
          'tah', 'tay', 'tee', 'toh', 'too', 'gah', 'gay', 'gee', 'goh', 'goo'
        ],
        answers: [['t', 'g']],
      },
      {
        title: 't/j',
        stimuli: [
          'tah', 'tay', 'tee', 'toh', 'too', 'yah', 'yay', 'yee', 'yoh', 'yoo'
        ],
        answers: [['t', 'j']],
      },
    ],
  },
  {
    sessionName: 'Discrimination - Group 4',
    tasks: discriminationProtocol,
    pairs: [
      {
        title: 'd/n',
        stimuli: [
          'dah', 'day', 'dee', 'doh', 'doo', 'nah', 'nay', 'nee', 'noh', 'noo'
        ],
        answers: [['d', 'n']],
      },
      {
        title: 'd/k',
        stimuli: [
          'dah', 'day', 'dee', 'doh', 'doo', 'kah', 'kay', 'kee', 'koh', 'koo'
        ],
        answers: [['d', 'k']],
      },
      {
        title: 'd/g',
        stimuli: [
          'dah', 'day', 'dee', 'doh', 'doo', 'gah', 'gay', 'gee', 'goh', 'goo'
        ],
        answers: [['d', 'g']],
      },
      {
        title: 'd/j',
        stimuli: [
          'dah', 'day', 'dee', 'doh', 'doo', 'yah', 'yay', 'yee', 'yoh', 'yoo'
        ],
        answers: [['d', 'j']],
      },
      {
        title: 'n/k',
        stimuli: [
          'nah', 'nay', 'nee', 'noh', 'noo', 'kah', 'kay', 'kee', 'koh', 'koo'
        ],
        answers: [['n', 'k']],
      },
    ],
  },
  {
    sessionName: 'Discrimination - Group 5',
    tasks: discriminationProtocol,
    pairs: [
      {
        title: 'n/g',
        stimuli: [
          'nah', 'nay', 'nee', 'noh', 'noo', 'gah', 'gay', 'gee', 'goh', 'goo'
        ],
        answers: [['n', 'g']],
      },
      {
        title: 'n/j',
        stimuli: [
          'nah', 'nay', 'nee', 'noh', 'noo', 'yah', 'yay', 'yee', 'yoh', 'yoo'
        ],
        answers: [['n', 'j']],
      },
      {
        title: 'k/g',
        stimuli: [
          'kah', 'kay', 'kee', 'koh', 'koo', 'gah', 'gay', 'gee', 'goh', 'goo'
        ],
        answers: [['k', 'g']],
      },
      {
        title: 'k/j',
        stimuli: [
          'kah', 'kay', 'kee', 'koh', 'koo', 'yah', 'yay', 'yee', 'yoh', 'yoo'
        ],
        answers: [['k', 'j']],
      },
      {
        title: 'g/j',
        stimuli: [
          'gah', 'gay', 'gee', 'goh', 'goo', 'yah', 'yay', 'yee', 'yoh', 'yoo'
        ],
        answers: [['g', 'j']],
      },
    ],
  },
  {
    sessionName: 'Discrimination - Group 6',
    tasks: discriminationProtocol,
    pairs: [
      {
        title: 'i/ɪ',
        stimuli: ['heed', 'hid'],
        answers: [['heed', 'hid']],
      },
      {
        title: 'e/eɪ',
        stimuli: ['head', 'hayed'],
        answers: [['head', 'hayed']],
      },
      {
        title: 'a/aɪ',
        stimuli: ['hod', 'hide'],
        answers: [['hod', 'hide']],
      },
      {
        title: 'ɑ/ɔɪ',
        stimuli: ['hawed', 'hoyed'],
        answers: [['hawed', 'hoyed']],
      },
    ],
  },
  {
    sessionName: 'Identification',
    tasks: identificationProtocol,
    pairs: [
      {
        title: 'Consonants',
        stimuli: [
          'bah',  'bee',  'boo',  'chah', 'chee', 'choo', 'dah', 'dee',
          'doo',  'dhah', 'dhee', 'dhoo', 'fah',  'fee',  'foo', 'gah',
          'gee',  'goo',  'jah',  'jee',  'joo',  'kah',  'kee', 'koo',
          'lah',  'lee',  'loo',  'mah',  'mee',  'moo',  'nah', 'nee',
          'noo',  'pah',  'pee',  'poo',  'rah',  'ree',  'roo', 'sah',
          'see',  'soo',  'shah', 'shee', 'shoo', 'tah',  'tee', 'too',
          'thah', 'thee', 'thoo', 'vah',  'vee',  'voo',  'wah', 'wee',
          'woo',  'yah',  'yee',  'yoo',  'zah',  'zee',  'zoo', 'zhah',
          'zhee', 'zhoo',
        ],
        answers: [[
          'b', 'ch', 'd', 'dh', 'f', 'g',  'j', 'k', 'l', 'm', 'n',
          'p', 'r',  's', 'sh', 't', 'th', 'v', 'w', 'y', 'z', 'zh'
        ]],
      },
      {
        title: 'Vowels',
        stimuli: [
          'had', 'hawed', 'hayed', 'head', 'heard', 'heed', 'hid', 'hide',
          'hod', 'hoed', 'hood', 'how\'d', 'hoyed', 'hud', 'who\'d'
        ],
        answers: [[
          'had', 'hawed', 'hayed', 'head', 'heard', 'heed', 'hid', 'hide',
          'hod', 'hoed', 'hood', 'how\'d', 'hoyed', 'hud', 'who\'d'
        ]],
      },
    ],
  },
  {
    sessionName: 'Structured Sentences - Training - L',
    tasks: structuredSentencesTrainLProtocol,
    pairs: [
      {
        title: 'List 1',
        stimuli: [
          'sentence_77', 'sentence_35', 'sentence_54', 'sentence_sentence_5',
          'sentence_10', 'sentence_251', 'sentence_78', 'sentence_464',
          'sentence_187', 'sentence_392'
        ],
        answers: [
          [
            'Peter', 'Kathy', 'Lucy', 'Allen', 'Rachel', 'William', 'Steven',
            'Thomas', 'Doris', 'Nina'
          ],
          [
            'got', 'sees', 'bought', 'gives', 'sold', 'prefers', 'has', 'kept',
            'ordered', 'wants'
          ],
          [
            'three', 'nine', 'seven', 'eight', 'four', 'nineteen', 'two',
            'fifteen', 'twelve', 'sixty'
          ],
          [
            'large', 'small', 'old', 'dark', 'heavy', 'green', 'cheap',
            'pretty', 'red', 'white'
          ],
          [
            'desks', 'chairs', 'tables', 'toys', 'spoons', 'windows', 'sofas',
            'rings', 'flowers', 'houses'
          ]
        ],
      },
    ],
  },
  {
    sessionName: 'Structured Sentences - Training - L + T',
    tasks: structuredSentencesTrainLTProtocol,
    pairs: [
      {
        title: 'List 2',
        stimuli: [
          'sentence_396', 'sentence_300', 'sentence_72', 'sentence_60',
          'sentence_167', 'sentence_19', 'sentence_320', 'sentence_36',
          'sentence_100', 'sentence_351'
        ],
        answers: [
          [
            'Peter', 'Kathy', 'Lucy', 'Allen', 'Rachel', 'William', 'Steven',
            'Thomas', 'Doris', 'Nina'
          ],
          [
            'got', 'sees', 'bought', 'gives', 'sold', 'prefers', 'has', 'kept',
            'ordered', 'wants'
          ],
          [
            'three', 'nine', 'seven', 'eight', 'four', 'nineteen', 'two',
            'fifteen', 'twelve', 'sixty'
          ],
          [
            'large', 'small', 'old', 'dark', 'heavy', 'green', 'cheap',
            'pretty', 'red', 'white'
          ],
          [
            'desks', 'chairs', 'tables', 'toys', 'spoons', 'windows', 'sofas',
            'rings', 'flowers', 'houses'
          ]
        ],
      },
    ],
  },
  {
    sessionName: 'Structured Sentences - Testing - L',
    tasks: structuredSentencesTestLProtocol,
    pairs: [
      {
        title: 'List 3',
        stimuli: [
          'sentence_497', 'sentence_130', 'sentence_83', 'sentence_191',
          'sentence_21', 'sentence_40', 'sentence_295', 'sentence_16',
          'sentence_153', 'sentence_446'
        ],
        answers: [
          [
            'Peter', 'Kathy', 'Lucy', 'Allen', 'Rachel', 'William', 'Steven',
            'Thomas', 'Doris', 'Nina'
          ],
          [
            'got', 'sees', 'bought', 'gives', 'sold', 'prefers', 'has', 'kept',
            'ordered', 'wants'
          ],
          [
            'three', 'nine', 'seven', 'eight', 'four', 'nineteen', 'two',
            'fifteen', 'twelve', 'sixty'
          ],
          [
            'large', 'small', 'old', 'dark', 'heavy', 'green', 'cheap',
            'pretty', 'red', 'white'
          ],
          [
            'desks', 'chairs', 'tables', 'toys', 'spoons', 'windows', 'sofas',
            'rings', 'flowers', 'houses'
          ]
        ],
      },
      {
        title: 'List 4',
        stimuli: [
          'sentence_98', 'sentence_259', 'sentence_338', 'sentence_273',
          'sentence_101', 'sentence_143', 'sentence_421', 'sentence_366',
          'sentence_476', 'sentence_313'
        ],
        answers: [
          [
            'Peter', 'Kathy', 'Lucy', 'Allen', 'Rachel', 'William', 'Steven',
            'Thomas', 'Doris', 'Nina'
          ],
          [
            'got', 'sees', 'bought', 'gives', 'sold', 'prefers', 'has', 'kept',
            'ordered', 'wants'
          ],
          [
            'three', 'nine', 'seven', 'eight', 'four', 'nineteen', 'two',
            'fifteen', 'twelve', 'sixty'
          ],
          [
            'large', 'small', 'old', 'dark', 'heavy', 'green', 'cheap',
            'pretty', 'red', 'white'
          ],
          [
            'desks', 'chairs', 'tables', 'toys', 'spoons', 'windows', 'sofas',
            'rings', 'flowers', 'houses'
          ]
        ],
      },
      {
        title: 'List 5',
        stimuli: [
          'sentence_49', 'sentence_238', 'sentence_498', 'sentence_120',
          'sentence_365', 'sentence_129', 'sentence_41', 'sentence_488',
          'sentence_390', 'sentence_420'
        ],
        answers: [
          [
            'Peter', 'Kathy', 'Lucy', 'Allen', 'Rachel', 'William', 'Steven',
            'Thomas', 'Doris', 'Nina'
          ],
          [
            'got', 'sees', 'bought', 'gives', 'sold', 'prefers', 'has', 'kept',
            'ordered', 'wants'
          ],
          [
            'three', 'nine', 'seven', 'eight', 'four', 'nineteen', 'two',
            'fifteen', 'twelve', 'sixty'
          ],
          [
            'large', 'small', 'old', 'dark', 'heavy', 'green', 'cheap',
            'pretty', 'red', 'white'
          ],
          [
            'desks', 'chairs', 'tables', 'toys', 'spoons', 'windows', 'sofas',
            'rings', 'flowers', 'houses'
          ]
        ],
      },
      {
        title: 'List 6',
        stimuli: [
          'sentence_144', 'sentence_164', 'sentence_472', 'sentence_105',
          'sentence_111', 'sentence_422', 'sentence_156', 'sentence_284',
          'sentence_430', 'sentence_382'
        ],
        answers: [
          [
            'Peter', 'Kathy', 'Lucy', 'Allen', 'Rachel', 'William', 'Steven',
            'Thomas', 'Doris', 'Nina'
          ],
          [
            'got', 'sees', 'bought', 'gives', 'sold', 'prefers', 'has', 'kept',
            'ordered', 'wants'
          ],
          [
            'three', 'nine', 'seven', 'eight', 'four', 'nineteen', 'two',
            'fifteen', 'twelve', 'sixty'
          ],
          [
            'large', 'small', 'old', 'dark', 'heavy', 'green', 'cheap',
            'pretty', 'red', 'white'
          ],
          [
            'desks', 'chairs', 'tables', 'toys', 'spoons', 'windows', 'sofas',
            'rings', 'flowers', 'houses'
          ]
        ],
      },
    ],
  },
  {
    sessionName: 'Structured Sentences - Testing - L + T',
    tasks: structuredSentencesTestLTProtocol,
    pairs: [
      {
        title: 'List 7',
        stimuli: [
          'sentence_248', 'sentence_70', 'sentence_468', 'sentence_374',
          'sentence_303', 'sentence_227', 'sentence_387', 'sentence_281',
          'sentence_56', 'sentence_260'
        ],
        answers: [
          [
            'Peter', 'Kathy', 'Lucy', 'Allen', 'Rachel', 'William', 'Steven',
            'Thomas', 'Doris', 'Nina'
          ],
          [
            'got', 'sees', 'bought', 'gives', 'sold', 'prefers', 'has', 'kept',
            'ordered', 'wants'
          ],
          [
            'three', 'nine', 'seven', 'eight', 'four', 'nineteen', 'two',
            'fifteen', 'twelve', 'sixty'
          ],
          [
            'large', 'small', 'old', 'dark', 'heavy', 'green', 'cheap',
            'pretty', 'red', 'white'
          ],
          [
            'desks', 'chairs', 'tables', 'toys', 'spoons', 'windows', 'sofas',
            'rings', 'flowers', 'houses'
          ]
        ],
      },
      {
        title: 'List 8',
        stimuli: [
          'sentence_318', 'sentence_310', 'sentence_404', 'sentence_243',
          'sentence_268', 'sentence_447', 'sentence_216', 'sentence_222',
          'sentence_106', 'sentence_242'
        ],
        answers: [
          [
            'Peter', 'Kathy', 'Lucy', 'Allen', 'Rachel', 'William', 'Steven',
            'Thomas', 'Doris', 'Nina'
          ],
          [
            'got', 'sees', 'bought', 'gives', 'sold', 'prefers', 'has', 'kept',
            'ordered', 'wants'
          ],
          [
            'three', 'nine', 'seven', 'eight', 'four', 'nineteen', 'two',
            'fifteen', 'twelve', 'sixty'
          ],
          [
            'large', 'small', 'old', 'dark', 'heavy', 'green', 'cheap',
            'pretty', 'red', 'white'
          ],
          [
            'desks', 'chairs', 'tables', 'toys', 'spoons', 'windows', 'sofas',
            'rings', 'flowers', 'houses'
          ]
        ],
      },
      {
        title: 'List 9',
        stimuli: [
          'sentence_125', 'sentence_74', 'sentence_28', 'sentence_159',
          'sentence_121', 'sentence_128', 'sentence_333', 'sentence_68',
          'sentence_196', 'sentence_280'
        ],
        answers: [
          [
            'Peter', 'Kathy', 'Lucy', 'Allen', 'Rachel', 'William', 'Steven',
            'Thomas', 'Doris', 'Nina'
          ],
          [
            'got', 'sees', 'bought', 'gives', 'sold', 'prefers', 'has', 'kept',
            'ordered', 'wants'
          ],
          [
            'three', 'nine', 'seven', 'eight', 'four', 'nineteen', 'two',
            'fifteen', 'twelve', 'sixty'
          ],
          [
            'large', 'small', 'old', 'dark', 'heavy', 'green', 'cheap',
            'pretty', 'red', 'white'
          ],
          [
            'desks', 'chairs', 'tables', 'toys', 'spoons', 'windows', 'sofas',
            'rings', 'flowers', 'houses'
          ]
        ],
      },
      {
        title: 'List 10',
        stimuli: [
          'sentence_170', 'sentence_389', 'sentence_289', 'sentence_110',
          'sentence_491', 'sentence_329', 'sentence_343', 'sentence_433',
          'sentence_410', 'sentence_210'
        ],
        answers: [
          [
            'Peter', 'Kathy', 'Lucy', 'Allen', 'Rachel', 'William', 'Steven',
            'Thomas', 'Doris', 'Nina'
          ],
          [
            'got', 'sees', 'bought', 'gives', 'sold', 'prefers', 'has', 'kept',
            'ordered', 'wants'
          ],
          [
            'three', 'nine', 'seven', 'eight', 'four', 'nineteen', 'two',
            'fifteen', 'twelve', 'sixty'
          ],
          [
            'large', 'small', 'old', 'dark', 'heavy', 'green', 'cheap',
            'pretty', 'red', 'white'
          ],
          [
            'desks', 'chairs', 'tables', 'toys', 'spoons', 'windows', 'sofas',
            'rings', 'flowers', 'houses'
          ]
        ],
      },
    ],
  },
  {
    sessionName: 'CUNY Sentences - Training - L',
    tasks: freeSentencesTrainLProtocol,
    pairs: [
      {
        title: 'List 1',
        stimuli: [
          'cuny_1_1', 'cuny_1_2', 'cuny_1_3', 'cuny_1_4', 'cuny_1_5',
          'cuny_1_6', 'cuny_1_7', 'cuny_1_8', 'cuny_1_9', 'cuny_1_10',
          'cuny_1_11', 'cuny_1_12'
        ],
        answers: [[]],
      },
    ],
  },
  {
    sessionName: 'CUNY Sentences - Training - L+T',
    tasks: freeSentencesTrainLTProtocol,
    pairs: [
      {
        title: 'List 2',
        stimuli: [
          'cuny_2_1', 'cuny_2_2', 'cuny_2_3', 'cuny_2_4', 'cuny_2_5',
          'cuny_2_6', 'cuny_2_7', 'cuny_2_8', 'cuny_2_9', 'cuny_2_10',
          'cuny_2_11', 'cuny_2_12'
        ],
        answers: [[]],
      },
    ],
  },
  {
    sessionName: 'CUNY Sentences - Testing - L',
    tasks: freeSentencesTestLProtocol,
    pairs: [
      {
        title: 'List 3',
        stimuli: [
          'cuny_3_1', 'cuny_3_2', 'cuny_3_3', 'cuny_3_4', 'cuny_3_5',
          'cuny_3_6', 'cuny_3_7', 'cuny_3_8', 'cuny_3_9', 'cuny_3_10',
          'cuny_3_11', 'cuny_3_12'
        ],
        answers: [[]],
      },
      {
        title: 'List 4',
        stimuli: [
          'cuny_4_1', 'cuny_4_2', 'cuny_4_3', 'cuny_4_4', 'cuny_4_5',
          'cuny_4_6', 'cuny_4_7', 'cuny_4_8', 'cuny_4_9', 'cuny_4_10',
          'cuny_4_11', 'cuny_4_12'
        ],
        answers: [[]],
      },
      {
        title: 'List 5',
        stimuli: [
          'cuny_5_1', 'cuny_5_2', 'cuny_5_3', 'cuny_5_4', 'cuny_5_5',
          'cuny_5_6', 'cuny_5_7', 'cuny_5_8', 'cuny_5_9', 'cuny_5_10',
          'cuny_5_11', 'cuny_5_12'
        ],
        answers: [[]],
      },
      {
        title: 'List 6',
        stimuli: [
          'cuny_6_1', 'cuny_6_2', 'cuny_6_3', 'cuny_6_4', 'cuny_6_5',
          'cuny_6_6', 'cuny_6_7', 'cuny_6_8', 'cuny_6_9', 'cuny_6_10',
          'cuny_6_11', 'cuny_6_12'
        ],
        answers: [[]],
      },
    ],
  },
  {
    sessionName: 'CUNY Sentences - Testing - L+T',
    tasks: freeSentencesTestLTProtocol,
    pairs: [
      {
        title: 'List 7',
        stimuli: [
          'cuny_7_1', 'cuny_7_2', 'cuny_7_3', 'cuny_7_4', 'cuny_7_5',
          'cuny_7_6', 'cuny_7_7', 'cuny_7_8', 'cuny_7_9', 'cuny_7_10',
          'cuny_7_11', 'cuny_7_12'
        ],
        answers: [[]],
      },
      {
        title: 'List 8',
        stimuli: [
          'cuny_8_1', 'cuny_8_2', 'cuny_8_3', 'cuny_8_4', 'cuny_8_5',
          'cuny_8_6', 'cuny_8_7', 'cuny_8_8', 'cuny_8_9', 'cuny_8_10',
          'cuny_8_11', 'cuny_8_12'
        ],
        answers: [[]],
      },
      {
        title: 'List 9',
        stimuli: [
          'cuny_9_1', 'cuny_9_2', 'cuny_9_3', 'cuny_9_4', 'cuny_9_5',
          'cuny_9_6', 'cuny_9_7', 'cuny_9_8', 'cuny_9_9', 'cuny_9_10',
          'cuny_9_11', 'cuny_9_12'
        ],
        answers: [[]],
      },
      {
        title: 'List 10',
        stimuli: [
          'cuny_10_1', 'cuny_10_2', 'cuny_10_3', 'cuny_10_4', 'cuny_10_5',
          'cuny_10_6', 'cuny_10_7', 'cuny_10_8', 'cuny_10_9', 'cuny_10_10',
          'cuny_10_11', 'cuny_10_12'
        ],
        answers: [[]],
      },
    ],
  },
];
