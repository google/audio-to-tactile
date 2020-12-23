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

"""Code to run psychophysics tests using a Jupyter notebook.

This code implements the GUI for 2 alternative forced choiced experiments, using
either audio (for easy testing) or tactors.
"""

import json
import random

import IPython.display as display
import ipywidgets as widgets
import levitt_experiment as levitt
import numpy as np

import sleeve_usb

# Basic code from:
#  https://stackoverflow.com/questions/54813069/python-onclick-button-widget-return-object-


class TestGui(object):
  """Basic Jupyter GUI for a psychophysical test.

  Just play as many beeps as the trial number, which goes up each time
  you click an answer button.
  """

  def __init__(
      self,
      button_names=('First', 'Second'),
      title='This is the Experiment Title',
      description='Click on the button indicating which half is higher.'):
    self.button_names = button_names
    self.title = title
    self.description = description

    self.signal = None
    self.fs = 16000.0
    self.trial_num = 1
    self.stimulus_pane = widgets.Output()

  def experiment_parameters(self):
    return {'fs': self.fs}

  def create_widgets(self):
    """Creates the ipython widgets needed for the display.  Called once."""
    self.title_pane = widgets.Label(
        self.title, disabled=False, style={'font_weight': 'bold'})
    self.description_pane = widgets.Label(self.description, disabled=False)

    self.button_widgets = {}
    for l in self.button_names:
      b = widgets.Button(
          description=l,
          disabled=False,
          button_style='warning',  # 'success', 'info', 'warning', 'danger', ''
          tooltip=f'{l} answer',
          icon='check'  # (FontAwesome names without the `fa-` prefix)
      )
      b.on_click(self.answer_event_handler)
      self.button_widgets[l] = b

    self.legend_pane = widgets.Label(
        f'Trial number {self.trial_num}', disabled=False)
    self.debug_pane = widgets.Label('', disabled=False)

  def display_widgets(self):
    """Displays the widgets in the output panel.  Called once."""
    self.create_widgets()
    self.create_stimulus()
    answer_pane = widgets.HBox(list(self.button_widgets.values()))
    self.show_stimulus(autoplay=False)
    display.display(
        widgets.VBox([
            self.title_pane,
            self.description_pane,
            self.stimulus_pane,
            answer_pane,
            self.legend_pane,
            self.debug_pane,
        ]))

  def update_display(self, new_legend):
    """Updates the display after each trial."""
    self.legend_pane.value = new_legend
    self.show_stimulus()

  def update_debug(self, new_message: str):
    """Updates the display after each trial."""
    self.debug_pane.value = new_message

  def create_stimulus(self):
    """Given the status of the experiment, creates the current stimulus.

    In this case, the audio consists of trial_num beeps, for testing code.
    """
    stim_len = 0.2
    self.fs = 16000
    blip_len = int(stim_len * self.fs)
    blip = np.zeros(blip_len)
    t = np.arange(int(blip_len * .75)) / float(self.fs)
    blip[0:t.shape[0]] = np.sin(2 * np.pi * t * 440)
    self.test_signal = np.concatenate([blip for i in range(self.trial_num)],
                                      axis=0)
    self.update_debug(f'DEBUG: Now playing {self.trial_num} beeps.')

  def show_stimulus(self, done_message: str = None, autoplay=True):
    """Shows stimulus specific pane, perhaps updated after each trial."""
    with self.stimulus_pane:
      display.clear_output()
      if done_message:
        a = done_message
      else:
        a = display.Audio(
            data=self.test_signal, rate=self.fs, autoplay=autoplay)
      display.display(a)

  def answer_event_handler(self, obj):
    """Handles the button clicks, checks answer, shows next trial."""
    print(f'Got a click from {obj.description}.')
    self.check_answer(obj.description)
    self.show_next_trial()

  def show_next_trial(self):
    """Updates the trial number count, and then updates the exp display."""
    self.trial_num += 1
    self.create_stimulus()
    self.update_display(f'Trial number {self.trial_num}')

  def check_answer(self, _):
    print('Generic check_answer called, probably an error.')
    pass  # Nothing to do for generic method.  Either answer is right


class Exp2AFC(TestGui):
  """Defined and implements a 2 alternative, forced choice test."""

  def __init__(self, max_runs=8, initial_level=0.05, **kwargs):
    """Creates the experiment.

    Just a dummy audio experiment for testing.

    Specialize this to do a real experiment, hopefully these methods will
    make that easier.

    Args:
      max_runs: How many runs to test the subject on. (A run is a
        monotonic sequence of level changes, as defined by Levitt.)
      initial_level: Which stimulus level to start the experiment with
        (meaning depends on the experiment.)
      **kwargs: keyword arguments for the super class.
    """
    self.max_runs = max_runs
    self.test_level = initial_level

    self.levitt_exp = levitt.LevittExp(
        initial_level=initial_level,
        change_delta=0.5,
        decrease_step_by_run=True,
        multiplicative_step=True)

    self.correct_answer = None
    self.last_result = ''
    self.test_description = ''
    super().__init__(**kwargs)

  def experiment_parameters(self):
    return {
        'max_runs': self.max_runs,
        'initial_level': self.initial_levels,
        'levitt_results': self.levitt_exp.results(),
        'levitt_threshold': self.levitt_exp.calculate_threshold(),
        'type': str(type(self)),
        'button_names': self.button_names,
        'title': self.title,
        'description': self.description,
    }

  def check_answer(self, trial_answer: str):
    """Checks to see if the right answer was received and updates the Levitt parameters."""
    correct = self.correct_answer in trial_answer.lower()
    if correct:
      self.last_result = 'Correct'
    else:
      self.last_result = 'Incorrect'
    self.test_level = self.levitt_exp.note_response(correct)
    print('Check_answer got %s, responding with level %g' %
          (correct, self.test_level))

  def show_next_trial(self):
    """Shows an experimental trial.

    Checks to see if we have done enough runs, then exits.
    Otherwise, creates the audio GUI widget and displays it for the
    subject's action.
    """
    if self.levitt_exp.run_number > self.max_runs:
      self.test_level = self.levitt_exp.calculate_threshold()
      self.show_stimulus('All done, with a threshold of %g.' % self.test_level)
      return
    super().show_next_trial()
    msg = f'Last result was {self.last_result.lower()}. '
    msg += f'Now showing run #{self.levitt_exp.run_number}, '
    msg += f'trial #{self.levitt_exp.trial_number}. '
    msg += f'This test is {self.test_description}.'
    self.update_debug(msg)

  def save_experiment(self, filename):
    exp_dict = self.experiment_parameters()
    with open(filename, 'w') as fp:
      json.dump(exp_dict, fp)


class AudioExp2AFC(Exp2AFC):
  """Do a simple pitch-based 2AFC test, just for testing."""

  def __init__(self, fs=16000, f0=440, **kwargs):
    self.fs = fs
    self.f0 = f0
    self.blip_len = 0.5
    self.pitch_std = 0.5
    super().__init__(**kwargs)

  def experiment_parameters(self):
    params = super().experiment_parameters()
    params['fs'] = self.fs
    params['f0'] = self.f0
    params['blip_len'] = self.blip_len
    params['pitch_std'] = self.pitch_std
    return params

  def create_stimulus(self) -> None:
    """Creates a 2 alternative forced choice pitch JND experiment.

    Stores for later use:
      The NP array of mono data at the desired sample rate, and
      a boolean that tells which segment (first or second) is higher.
    """
    if self.test_level < 0:
      raise ValueError('Difference argument should be > 0, not %s' %
                       self.test_level)
    f1 = random.normalvariate(self.f0, self.pitch_std)
    f2 = (1 + self.test_level) * f1  # Always higher
    if random.random() < 0.5:
      self.correct_answer = 'first'
      s1 = f2
      s2 = f1
    else:
      self.correct_answer = 'second'
      s1 = f1
      s2 = f2

    stim_len = int(self.fs * self.blip_len)
    t = np.arange(2 * stim_len) / float(self.fs)
    window = np.hanning(stim_len)
    self.test_signal = 0 * t
    self.test_signal[:stim_len] = np.sin(2 * np.pi * s1 * t[:stim_len]) * window
    self.test_signal[stim_len:] = np.sin(2 * np.pi * s2 * t[stim_len:]) * window
    self.test_description = '%g -> %g with step %g' % (s1, s2, self.test_level)


class TactorExp2AFC(Exp2AFC):
  """Tactor amplitude threshold level experiment."""

  def __init__(self,
               f0: float = 50,
               blip_len: float = 0.5,
               stim_channel: int = 0,
               click_channel: int = 3,
               mask_channel: int = 2,
               mask_level: float = 0,
               **kwargs):
    """Initialize a tactor experiment.

    Args:
        f0: frequency of the buzz
        blip_len: Length of each sample (1/2 of total) in seconds
        stim_channel: Which channel on the sleeve is being tested.
        click_channel: Which channel gets the click indicating the center
          point.
        mask_channel: Where to put the noise mask signal.
        mask_level: Amplitude of the masking signal.  Zero means no mask.
        **kwargs: Arguments for the super class.
    """
    self.f0 = f0
    self.blip_len = blip_len
    self.stim_channel = stim_channel
    self.click_channel = click_channel
    self.mask_channel = mask_channel
    self.mask_level = mask_level

    self.sleeve = sleeve_usb.SleeveUSB()
    self.fs = self.sleeve.SAMPLE_RATE

    super().__init__(**kwargs)

  def experiment_parameters(self):
    params = super().experiment_parameters()
    params['f0'] = self.f0
    params['blip_len'] = self.blip_len
    params['stim_channel'] = self.stim_channel
    params['click_channel'] = self.click_channel
    params['mask_channel'] = self.mask_channel
    params['mask_level'] = self.mask_level
    return params

  def create_stimulus(self) -> None:
    """Creates a 2 alternative forced choice tactor threshold experiment.

    Computes: Sets two items, the NP array of tactor data, and a boolean
      that tells which segment (first or second) is higher.
    """
    if self.test_level < 0:
      raise ValueError('Level argument should be > 0, not %s' % self.test_level)
    fs = self.sleeve.SAMPLE_RATE

    blip_len = int(self.blip_len * fs)
    test_stim = np.zeros((blip_len, self.sleeve.TACTILE_CHANNELS))

    t = np.arange(blip_len) / float(fs)
    window = np.hanning(blip_len)
    test_stim[:, self.stim_channel] = self.test_level * np.sin(
        2 * np.pi * self.f0 * t) * window

    if random.random() < 0.5:
      self.correct_answer = 'first'
      s1 = test_stim
      s2 = 0 * test_stim
    else:
      self.correct_answer = 'second'
      s1 = 0 * test_stim
      s2 = test_stim

    self.test_signal = np.concatenate((s1, s2), axis=0)

    if self.mask_level > 0:
      np.clip(self.mask_level * np.random.standard_normal(2 * blip_len), -1, 1,
              self.test_signal[:, self.mask_channel])
    # Add clicks to another channel so user can orient.
    click_len = 50
    click = np.sin(2 * np.pi * 250 * t[:click_len])
    # self.signal[:click_len, click_channel] = click
    self.test_signal[blip_len:(blip_len + click_len),
                     self.click_channel] = click
    # self.signal[-click_len:, click_channel] = click
    self.test_description = 'Stimulus level %g is in the %s segment (%d & %d)' % (
        self.test_level, self.correct_answer, self.stim_channel,
        self.click_channel)

  def _play_stimulus(self, _=None):
    sleeve = sleeve_usb.SleeveUSB()
    sleeve.send_waves_to_tactors(self.test_signal)

  def show_stimulus(self, done_message: str = None, autoplay=True):
    """Displays the UI that presents the stimulus, audio in this case."""
    del autoplay
    with self.stimulus_pane:
      display.clear_output()
      if done_message:
        b = done_message
      else:
        b = widgets.Button(
            description='Play Stimulus',
            disabled=False,
            button_style='success',  # success, info, warning, danger, ''
            tooltip='play stimulus',
            icon='play'  # (FontAwesome names without the `fa-` prefix)
        )
        b.on_click(self._play_stimulus)
      display.display(b)


class TactorPhaseExp(TactorExp2AFC):
  """Test the phase sensitivity of our tactile sensors."""

  def __init__(self,
               title='Are the two signals the same or different?',
               button_names=('Same', 'Different'),
               initial_level=180.0,
               stim_level=0.75,
               stim2_channel=7,
               **kwargs):
    """Initialize the object with the experimental parameters.

    Args:
      title: Redefines title with new default.
      button_names: Redefines button names with new default.
      initial_level: Redefines level with new default (180 degrees.)
      stim_level: What amplitude to give the stimulus.
      stim2_channel: Which channel to use for second stimulus
      **kwargs: Arguments for the super class.
    """
    self.stim_level = stim_level
    self.stim2_channel = stim2_channel

    self.sleeve = sleeve_usb.SleeveUSB()
    self.fs = self.sleeve.SAMPLE_RATE
    super().__init__(
        title=title,
        initial_level=initial_level,
        button_names=button_names,
        **kwargs)

  def create_stimulus(self):
    """Creates a phase-difference tactor threshold experiment.

    Returns output in the class' test_signal and test_description slots.
    """
    if self.stim_level < 0:
      raise ValueError('Level argument should be > 0, not %s' % self.stim_level)
    if random.random() < 0.5:
      self.correct_answer = 'same'
      phase = 0
    else:
      self.correct_answer = 'different'
      phase = 2 * np.pi / 360.0 * max(0.0, min(180.0, self.stim_level))

    self.test_signal = self.synthesize_signal(self.f0, phase)

    if self.mask_level > 0:
      np.clip(self.mask_level * np.random.standard_normal(2 * self.blip_len),
              -1, 1,
              self.test_signal[:, self.mask_channel])

    self.test_description = 'Stimulus level %g is %s (%d & %d)' % (
        self.stim_level, self.correct_answer, self.stim_channel,
        self.stim2_channel)

  def experiment_parameters(self):
    params = super().experiment_parameters()
    params['initial_level'] = self.initial_level
    params['stim_level'] = self.stim_level
    params['stim2_channel'] = self.stim2_channel
    return params

  # The following methods define a simple GUI to let a user explore the
  # frequency phase space.

  def synthesize_signal(self, f0, phase):
    """Just synthesize one test signal: two channels at frequency and phase difference.

    Note: Several parameters of the signal are defined by the class when it
    is initialized,
    including blip_len, stim_level, stim_channel, stim2_channel.

    Args:
      f0: Frequency of the sinusoid.
      phase: Initial phase of the sinusoid in degrees.

    Returns:
      A multidimensional vector containing the desired signals.
    """
    fs = self.sleeve.SAMPLE_RATE

    blip_len = int(self.blip_len * fs)
    test_stim = np.zeros((blip_len, self.sleeve.TACTILE_CHANNELS))

    t = np.arange(blip_len) / float(fs)
    window = np.hanning(blip_len)

    test_stim[:, self.stim_channel] = self.stim_level * np.sin(
        2 * np.pi * f0 * t) * window
    test_stim[:, self.stim2_channel] = self.stim_level * np.sin(
        2 * np.pi * f0 * t + phase) * window
    return test_stim

  def play_event_handler(self, obj):
    """Handles the play_widget GUI, playing the desired stimulus."""
    f0 = self.f0_widget.value
    phase = self.phase_widget.value
    print(
        f'Got a click from {obj.description} for {f0}Hz at {phase} degrees.')
    self.test_signal = self.synthesize_signal(f0, phase)
    self._play_stimulus()

  def play_widget(self):
    """Creates a widget that lets us test different frequencies and phases."""
    self.f0_widget = widgets.FloatSlider(
        value=32,
        min=25,
        max=250,
        step=1,
        description='F0:',
        disabled=False,
        continuous_update=False,
        orientation='vertical',
        readout=True,
        readout_format='.1f',
    )

    self.phase_widget = widgets.FloatSlider(
        value=90,
        min=0,
        max=180.0,
        step=1,
        description='Phase:',
        disabled=False,
        continuous_update=False,
        orientation='vertical',
        readout=True,
        readout_format='.1f',
    )

    play_same = widgets.Button(
        description='Play Same',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='check'  # (FontAwesome names without the `fa-` prefix)
    )
    play_same.on_click(self.play_event_handler)

    play_different = widgets.Button(
        description='Play Different',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='check'  # (FontAwesome names without the `fa-` prefix)
    )
    play_different.on_click(self.play_event_handler)

    button_pane = widgets.VBox([play_same, play_different])
    test_pane = widgets.HBox([self.f0_widget, self.phase_widget, button_pane])
    return test_pane
