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

"""Python code implementing the Levitt up-down transformed experiment protocol.

See Harry Levitt's paper for more details:
  http://bdml.stanford.edu/twiki/pub/Haptics/DetectionThreshold/psychoacoustics.pdf

Note, many of the examples here use the data from his Figure 4.
"""

import operator
from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np


class LevittExp(object):
  """Implement the basic Levitt transformed up-down experiment."""

  def __init__(self,
               initial_level: float,
               change_delta: float = 1,
               decrease_step_by_run: bool = False,
               multiplicative_step: float = False,
               debug: bool = False):
    """Implements an up-down experiment searching for a threshold.

    This class takes in true/false decisions by the user and changes the
    level for the next experiment.  After creating the class with the
    appropriate parameters, call note_response() with each new user action.
    This method notes the response, and changes the level based on the
    internal response_rule() method.  The new level is returned for use in
    the next trial.

    Args:
      initial_level: Initial value for the experiment.
      change_delta: How much to change the level after each trial. When
        multiplicative_step is False, then the current step is
        add/subtracted from the current level. When multiplicative_step is
        True, then the current level is multiplied/divided by 1 plus the current
        step.
      decrease_step_by_run: The current step changes after each run (a
        monotonic change in the level).  As Levitt suggests, the current
        step is the initial_step divided by the number of runs so far.
      multiplicative_step: When true, the delta value represents a
          multiplicative factor, while when False the steps are additive.
        debug: Prints some internal state after each trial.
    """
    if change_delta < 0:
      raise ValueError('Change_delta should be positive, not %g.' %
                       change_delta)
    if multiplicative_step and change_delta > 1:
      raise ValueError('Multiplicative change should be less than 1, not %g.' %
                       change_delta)
    if multiplicative_step and initial_level == 0:
      raise ValueError('Do not start multiplicative steps starting at 0.')
    self.initial_level = initial_level
    self.level = initial_level
    self.initial_change_delta = change_delta
    self.change_delta = change_delta
    self.run_trial_history = [[]]  # A nested history of trials (run x trial #)
    self.change_history = [
    ]  # Keep track of change events so we can delineate runs
    self.level_history = []  # History of all levels over time
    self.trial_number = 0
    self.decrease_step_by_run = decrease_step_by_run
    self.multiplicative_step = multiplicative_step
    self.debug = debug

  @property
  def run_number(self) -> int:
    return len(self.run_trial_history)

  def note_response(self, new_answer):
    """Records the response to a trial, adjusts history and sets new level."""
    self.level_history.append(self.level)
    self.run_trial_history[-1].append(new_answer)
    self._response_rule()
    self.trial_number += 1
    if self.debug:
      last_run = self.run_trial_history[-1]
      new_trials = len(last_run)
      print(f'After time {self.trial_number}, {new_trials}: '
            f'Answer was {new_answer}, level is now {self.level}')
    return self.level

  def _response_rule(self):
    """Decides whether we change the level based on history."""
    last_run = self.run_trial_history[-1]
    # pylint: disable=g-explicit-length-test  Think of this as a list.
    if len(last_run) > 0:
      # Change level based on result of last test, down after positive result.
      self._change_level_down(last_run[-1])

  def _change_level_down(self, down):
    """Changes the level of an experiment and updates run history."""
    if self.debug:
      print('  Changing level down:', down)
    if self.multiplicative_step:
      if down:
        self.level /= (1 + self.change_delta)
      else:
        self.level *= (1 + self.change_delta)
    else:
      if down:
        self.level -= self.change_delta
      else:
        self.level += self.change_delta
    self.change_history.append(down)

    # Check to see if we are starting a new run
    if len(self.change_history
          ) > 1 and self.change_history[-2] != self.change_history[-1]:
      # Start a new run since direction changed
      if self.debug:
        print('Starting a new run...')
      self.run_trial_history.append([])
      if self.decrease_step_by_run:
        self.change_delta = self.initial_change_delta / len(
            self.run_trial_history)

  def calculate_threshold(self):
    start, end = self.run_boundaries(every_nth=1)
    num_runs = 2 * (len(start) // 2)  # Even number of runs
    start_levels = [self.level_history[i] for i in start[-num_runs:]]
    end_levels = [self.level_history[i] for i in end[-num_runs:]]
    all_points = start_levels + end_levels
    return sum(all_points) / float(num_runs) / 2.0

  def run_boundaries(self, every_nth=2):
    """Finds the boundaries between runs, defined as a stretch of ups or downs.

    Args:
      every_nth: Return every n'th boundary (so we can get every other one
        for plotting.)

    Returns:
      start, end: the starting and ending index for each run.  Note, the
      end of one trial is the start of the next.
    """

    # Add alternating color background for each run
    run_lengths = [len(sublist) for sublist in self.run_trial_history]
    run_boundaries = np.cumsum([0] + run_lengths)
    start = [i - 1 for i in run_boundaries[1::every_nth]]
    end = [i - 1 for i in run_boundaries[2::every_nth]]
    return start, end

  def plot_response(self, title: str = '') -> None:
    """Create a graph to summarize an experiment.

    Plot each test and its result, show vertical bars for each run.

    Args:
      title: Text to put at the top fo the graph.
    """
    # pylint: disable=g-complex-comprehension   Unpack list of lists
    run_history = [
        item for sublist in self.run_trial_history for item in sublist
    ]

    # Plot positive and negative trials separately
    positives = [i for i, v in enumerate(run_history) if v]
    negatives = [i for i, v in enumerate(run_history) if not v]
    # https://stackoverflow.com/questions/18272160/access-multiple-elements-of-list-knowing-their-index
    plt.plot(positives,
             operator.itemgetter(*positives)(self.level_history), '+')
    plt.plot(negatives,
             operator.itemgetter(*negatives)(self.level_history), '*')

    start, end = self.run_boundaries()
    for s, e in zip(start, end):
      plt.gca().axvspan(s, e, zorder=0, alpha=0.25)
    plt.title(title)
    plt.xlabel('Trial Number')
    plt.ylabel('Experiment Level')

  def results(self) -> Dict[str, Any]:
    return {
        'run_boundaries': self.run_boundaries,
        'run_trial_history': self.run_trial_history,
        'level_history': self.level_history,
        'experimental_threshold': self.calculate_threshold(),
    }


class Levitt2down1upAdditive(LevittExp):
  """A modification of the basic Levitt experiment.

  Now decreases the level after *two* positive results, and goes up after
  just one.
  """

  def _response_rule(self):
    last_run = self.run_trial_history[-1]
    new_trials = len(last_run)
    # pylint: disable=g-explicit-length-test  Think of this as a list.
    if len(last_run) > 0 and not last_run[-1]:  # Made a mistake
      self._change_level_down(False)
    elif len(last_run) > 1 and new_trials > 1 and last_run[-1] and last_run[-2]:
      # Got two right answers
      self._change_level_down(True)
