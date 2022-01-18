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

"""Test code for Levitt experimental code."""

from absl.testing import absltest
import numpy as np
from extras.python.psycho import levitt_experiment


class LevittTest(absltest.TestCase):

  # Data from Figure 4 of Levitt's 1971 paper.
  figure4_data = [True, True,
                  False, False, False, False,
                  True, True,
                  False, False, False,
                  True, True, True, True, True,
                  False, False, False,
                  True, True,
                  False, False, False,
                  True]  # Extra result needed to end the run.

  def test_levitt(self):
    exp = levitt_experiment.LevittExp(0, 1, decrease_step_by_run=False)
    for r in self.figure4_data:
      exp.note_response(r)

    starts, ends = exp.run_boundaries(1)
    self.assertEqual(starts, [2, 6, 8, 11, 16, 19, 21, 24])
    self.assertEqual(ends, [6, 8, 11, 16, 19, 21, 24])

    self.assertEqual(exp.calculate_threshold(), 0.375)

  def test_levitt_decreasing_steps(self):
    exp = levitt_experiment.LevittExp(0, 1, decrease_step_by_run=True)
    for r in self.figure4_data:
      exp.note_response(r)

    self.assertEqual(exp.calculate_threshold(), -0.2699404761904763)
    steps = np.abs(np.asarray(exp.level_history[1:]) -
                   np.asarray(exp.level_history[:-1]))
    # pylint: disable=bad-whitespace
    expected = np.array(
        [1.        , 1.        , 1.        , 0.5       , 0.5       ,
         0.5       , 0.5       , 0.33333333, 0.33333333, 0.25      ,
         0.25      , 0.25      , 0.2       , 0.2       , 0.2       ,
         0.2       , 0.2       , 0.16666667, 0.16666667, 0.16666667,
         0.14285714, 0.14285714, 0.125     , 0.125])
    np.testing.assert_allclose(steps, expected, atol=1e-4)

  def test_levitt_multiplicative_steps(self):
    exp = levitt_experiment.LevittExp(1, 1,
                                      decrease_step_by_run=True,
                                      multiplicative_step=True)
    for r in self.figure4_data:
      exp.note_response(r)
    steps = np.abs(np.asarray(exp.level_history[1:]) /
                   np.asarray(exp.level_history[:-1]))
    # pylint: disable=bad-whitespace
    expected = np.array(
        [0.5       , 0.5       , 2.        , 1.5       , 1.5       ,
         1.5       , 0.66666667, 0.75      , 1.33333333, 1.25      ,
         1.25      , 0.8       , 0.83333333, 0.83333333, 0.83333333,
         0.83333333, 1.2       , 1.16666667, 1.16666667, 0.85714286,
         0.875     , 1.14285714, 1.125     , 1.125])
    np.testing.assert_allclose(steps, expected, atol=1e-4)

  def test_levitt_2_down_1_up(self):
    exp = levitt_experiment.Levitt2down1upAdditive(0, 1)
    for r in self.figure4_data:
      exp.note_response(r)

    steps = exp.level_history
    expected = [0, 0, -1, 0, 1, 2, 3, 3, 2, 3, 4, 5, 5, 4, 4, 3, 2,
                3, 4, 5, 5, 4, 5, 6, 7]
    self.assertEqual(steps, expected)

if __name__ == '__main__':
  absltest.main()
