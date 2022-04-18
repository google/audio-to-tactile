/**
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Tests the Levitt experiment protocol, based off data from Fig 4. Levitt 1970
 * and additional cases.
 */

goog.module('audio_to_tactile.extras.web.training.levittTest');
goog.setTestOnly();

const levitt = goog.require('audio_to_tactile.extras.web.training.levitt');
const testSuite = goog.require('goog.testing.testSuite');

/**
 * Tests a levitt experiment, checking level history, bounds, and threshold.
 * @param {!Array} data
 * @param {number} rule
 * @param {number} expectedThreshold
 * @param {!Array} expectedLevels
 * @param {!Array} expectedStartBounds
 * @param {!Array} expectedEndBounds
 */
function levittTest(data, rule, expectedThreshold, expectedLevels,
               expectedStartBounds, expectedEndBounds) {
    let exp = new levitt.LevittExperiment(
        0, 1, false, false, 8, false, -10, 10, rule);
    data.map((r) => {
      exp.noteResponse(r);
    });

    assertArrayEquals(expectedLevels, exp.levelHistory);

    let runBoundaries = exp.runBoundaries(2);
    assertArrayEquals(expectedStartBounds, runBoundaries[0]);
    assertArrayEquals(expectedEndBounds, runBoundaries[1]);

    let threshold = exp.calculateThreshold();
    assertEquals(expectedThreshold, threshold);
}

testSuite({
  testLevittFigure4() {
    const figure4Data = [true, true,
                          false, false, false, false,
                          true, true,
                          false, false, false,
                          true, true, true, true, true,
                          false, false, false,
                          true, true,
                          false, false, false,
                        ];

    let expectedThreshold = 0.25;
    let expectedLevels =
        [0,-1,-2,-1,0,1,2,1,0,1,2,3,2,1,0,-1,-2,-1,0,1,0,-1,0,1];
    let expectedStartBounds = [2, 8, 16, 21];
    let expectedEndBounds = [6, 11, 19, 23];

    levittTest(figure4Data, 1, expectedThreshold, expectedLevels,
               expectedStartBounds, expectedEndBounds);

  },
  testLevittExtraTrueResponse() {
    const data = [true, true,
                  false, false, false, false,
                  true, true,
                  false, false, false,
                  true, true, true, true, true,
                  false, false, false,
                  true, true,
                  false, false, false,
                  true,
                 ];

    let expectedThreshold = 0.375;
    let expectedLevels =
        [0,-1,-2,-1,0,1,2,1,0,1,2,3,2,1,0,-1,-2,-1,0,1,0,-1,0,1,2];
    let expectedStartBounds = [2, 8, 16, 21];
    let expectedEndBounds = [6, 11, 19, 24];

    levittTest(data, 1, expectedThreshold, expectedLevels,
               expectedStartBounds, expectedEndBounds);
  },
    testLevittTwoRuns() {
    const data = [true, true,
                  false, false, false, false,
                  true,
                 ];

    let expectedThreshold = 0;
    let expectedLevels = [0,-1,-2,-1,0,1,2];
    let expectedStartBounds = [2];
    let expectedEndBounds = [6];

    levittTest(data, 1, expectedThreshold, expectedLevels,
               expectedStartBounds, expectedEndBounds);
  },
  testLevittFourRuns() {
    const data = [true, true,
                  false, false, false, false,
                  true,true,
                  false,
                  true,
                 ];

    let expectedThreshold = 0.25;
    let expectedLevels = [0,-1,-2,-1,0,1,2,1,0,1];
    let expectedStartBounds = [2,8];
    let expectedEndBounds = [6,9];

    levittTest(data, 1, expectedThreshold, expectedLevels,
               expectedStartBounds, expectedEndBounds);
  },
  testLevittRule2() {
    const data = [true, true,
                  false, false, true, false, false,
                  true,true,
                  false,
                  true,
                 ];

    let expectedThreshold = 1.75;
    let expectedLevels = [0, 0, -1, 0, 1, 1, 2, 3, 3, 2, 3];
    let expectedStartBounds = [2, 9];
    let expectedEndBounds = [8,10];

    levittTest(data, 2, expectedThreshold, expectedLevels,
               expectedStartBounds, expectedEndBounds);
  }
});

