/**
 * Copyright 2021-2023 Google LLC
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
 * Implements a basic version of the Transformed Up-Down Methods in
 * Psychoacoustics (Levitt 1970), adapted from psycho/levitt_experiment.py, for
 * use with web-based experiment tools. Handles recording results and tracking
 * runs, as well as calculating final threshold values.
 */

class LevittExperiment {
  /**
   * Constructs an experiment in initial state.
   */
  constructor(
      initialLevel, changeDelta, decreaseStepByRun = false,
      multiplicativeStep = false, maxRuns = 16, debug = false, minLevel = 1,
      maxLevel = 63, responseRuleNumber = 1) {
    if (changeDelta <= 0 || changeDelta > 10) {
      console.error('Change delta must greater than 0, and <= 10.');
    }

    if ((maxRuns % 2) || maxRuns < 1) {
      console.error('Max runs must be an even positive integer.');
    }

    /** @type {number} */ this.initialLevel = initialLevel;
    /** @type {number} */ this.level = initialLevel;
    /** @type {number} */ this.initialChangeDelta = changeDelta;
    /** @type {number} */ this.changeDelta = changeDelta;
    // Nested history of trials (run x trial #)
    /** @type {!Array} */ this.runTrialHistory = [[]];
    // Track change events to delineate runs
    /** @type {!Array} */ this.changeHistory = [];
    // History of all levels over time
    /** @type {!Array} */ this.levelHistory = [];
    /** @type {number} */ this.trialNumber = 0;
    /** @type {boolean} */ this.decreaseStepByRun = decreaseStepByRun;
    /** @type {boolean} */ this.multiplicativeStep = multiplicativeStep;
    /** @type {number} */ this.maxRuns = maxRuns;
    /** @type {number} */ this.minLevel = minLevel;
    /** @type {number} */ this.maxLevel = maxLevel;
    /** @type {number} */ this.responseRuleNumber = responseRuleNumber;
    /** @type {boolean} */ this.debug = debug;
  }

  /**
   * Provides current number of runs (length of runTrialHistory).
   * @return {number} Current number of runs.
   */
  get runNumber() {
    return this.runTrialHistory.length;
  }

  /**
   * Indicates whether experiment is ongoing based on current run.
   * @return {boolean} Experiment status.
   */
  get incomplete() {
    return this.runNumber <= this.maxRuns;
  }

  /**
   * Indicates whether the current level has saturated to the min or max.
   * @return {boolean} True if saturated.
   */
  get saturated() {
    return !(this.minLevel < this.level && this.level < this.maxLevel);
  }

  /**
   * Records the response to a trial, adjusts history, and sets new level.
   * @param {boolean} newAnswer Correctness of the user response.
   * @return {number} New level value.
   */
  noteResponse(newAnswer) {
    this.levelHistory.push(this.level);
    this.runTrialHistory[this.runNumber - 1].push(newAnswer);
    this.responseRule();
    this.trialNumber += 1;
    if (this.debug) {
      let lastRun = this.runTrialHistory[this.runNumber - 1];
      let newTrials = lastRun.length;
      console.log(
          'After time ' + this.trialNumber + ', ' + newTrials + ': ' +
          ' Answer was ' + newAnswer + ', level is now ' + this.level);
    }
    return this.level;
  }

  /**
   * Defines the response rule for whether to change the level.
   */
  responseRule() {
    switch (this.responseRuleNumber) {
      case 1:
        this.rule1();
        break;
      case 2:
        this.rule2();
        break;
      default:
        console.log('No Rule Defined');
    }
  }

  /**
   * Rule Levitt_1:
   *    If the run has entries, adjust level up after one incorrect
   *    response (-), adjust level down after one correct response (+).
   */
  rule1() {
    let lastRun = this.runTrialHistory[this.runNumber - 1];
    if (lastRun.length > 0) {
      let responseResult = lastRun[lastRun.length - 1];
      this.changeLevelDown(responseResult);
    }
  }

  /**
   * Rule Levitt_2:
   *    If the run has entries, adjust level up after one incorrect
   *    response (+ - or -), adjust level down after two sequential correct
   *    responses (+ +).
   */
  rule2() {
    let lastRun = this.runTrialHistory[this.runNumber - 1];
    // incorrect reponse:
    if (lastRun.length > 0 && !lastRun[lastRun.length - 1]) {
      this.changeLevelDown(false);
    } else if (lastRun.length > 1 && (lastRun.length % 2 == 0) &&
          lastRun[lastRun.length - 1] && lastRun[lastRun.length - 2]) {
      this.changeLevelDown(true);
    }
  }

  /**
   * Determines whether a new run should start, by comparing the last answer
   * to the current answer. If they differ, or if the level has saturated, begin
   * a new run.
   * @return {boolean} Whether to start a new run.
   */
  beginNewRun() {
    let changeLength = this.changeHistory.length;
    let lastChange = this.changeHistory[changeLength - 2];
    let thisChange = this.changeHistory[changeLength - 1];
    return (changeLength > 1) && (lastChange != thisChange || this.saturated);
  }

  /**
   * Limits the acceptable gains to within minimum and maximum bounds;
   * displays a warning message if bounds are violated.
   */
  boundsCheckGain() {
    if (this.level > this.maxLevel) {
      console.error('MAXIMUM GAIN ERROR');
      this.level = this.maxLevel;
    } else if (this.level < this.minLevel) {
      console.error('MINIMUM GAIN ERROR');
      this.level = this.minLevel;
    }
  }

  /**
   * Changes the level of an experiment and updates run history.
   * @param {boolean} down Whether the level should change down.
   */
  changeLevelDown(down) {
    if (this.debug) {
      console.log('  Changing level ', down ? 'down' : 'up');
    }

    if (this.multiplicativeStep) {
      if (down) {
        this.level /= (1 + this.changeDelta);
      } else {
        this.level *= (1 + this.changeDelta);
      }
    } else {
      if (down) {
        this.level -= this.changeDelta;
      } else {
        this.level += this.changeDelta;
      }
    }
    this.boundsCheckGain();
    this.changeHistory.push(down);

    if (this.beginNewRun()) {
      if (this.debug) {
        console.log('*******Starting a new run...********');
      }
      this.runTrialHistory.push([]);
      if (this.decreaseStepByRun) {
        this.changeDelta =
            this.initialChangeDelta / this.runTrialHistory.length;
      }
    }
  }

  /**
   * Finds the boundaries between runs, defined as a stretch of ups or downs.
   * @param {number} everyNth Step size between runs of interest.  Use 2
   *    for calculating the threshold using midpoint estimation.
   * @return {!Array<!Array<number>>} First entry is an array of start indices;
   *    second entry is an array of end indices.
   */
  runBoundaries(everyNth) {
    let runLengths = this.runTrialHistory.map((run) => run.length);
    let runBoundaries = [0];
    for (let i = 0; i < runLengths.length; i++) {
      runBoundaries.push(runBoundaries[i] + runLengths[i]);
    }
    // Excludes the first run since it has an arbitrary initial point.
    let startIndices = runBoundaries.slice(1, -1)
                           .filter((e, i) => (i % everyNth) === 0)
                           .map((e) => e - 1);
    let endIndices = runBoundaries.slice(2)
                         .filter((e, i) => (i % everyNth) === 0)
                         .map((e) => e - 1);
    return [startIndices, endIndices];
  }

  /**
   * Calculates a final threshold value as the mean of the midpoints of
   * every second run, over an even number of runs.
   * (see Fig. 4, Levitt 1970)
   * @return {number} Threshold value.
   */
  calculateThreshold() {
    let runBoundaries = this.runBoundaries(2);
    let starts = runBoundaries[0];
    let ends = runBoundaries[1];
    let startLevels = starts.map((i) => this.levelHistory[i]);
    let endLevels = ends.map((i) => this.levelHistory[i]);
    let sumStarts = (startLevels.reduce((a, b) => a + b));
    let sumEnds = (endLevels.reduce((a, b) => a + b));
    let mean = (sumStarts + sumEnds) / starts.length / 2;
    return mean;
  }

  /**
   * Creates a readable string of all the experiment data.
   * @return {string} Experiment data for download or display.
   */
  getHistory() {
    let contents = '';
    contents += 'Trials: ' + this.runTrialHistory.toString() + '\n';
    contents += 'Levels: ' + this.levelHistory.toString() + '\n';
    return contents;
  }
}
