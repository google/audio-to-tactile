# Copyright 2022 Google LLC
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

"""Binary to run tap out receiver.

Connect the device with a USB cable and turn on the device. Then run this
program as

run_tap_out.py --port <portname> --capture <outputs> --duration <duration>

The <port> arg is the name where of the port where the device is connected,
<outputs> is a comma-separated list of outputs to capture, and <duration> is a
duration in seconds.

Run the program without --port to print a list of available ports.

Run the program without --capture to print a list of available outputs.

Captured data is written to --output, which defaults to a timestamp-suffixed
directory in your user home folder like ~/tap_out_YYYYMMDD_HHMMSS.
"""

import datetime
import os
import os.path
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

from absl import app
from absl import flags
import matplotlib.figure
import numpy as np
import serial
import serial.tools.list_ports

from extras.python import dsp
from extras.python import plot
from extras.python.tactile import tap_out

flags.DEFINE_string('port', None,
                    'Serial port where the device is connected. ' +
                    'If empty, prints a list of available ports.')

flags.DEFINE_list('capture', [],
                  'A comma-separated list of which tap_out outputs to ' +
                  'capture. If empty, prints a list of available outputs.')

flags.DEFINE_string('output', None, 'Output directory.')

flags.DEFINE_float('duration', 10.0, 'Recording duration in seconds.')

flags.DEFINE_bool('csv', False,
                  'If true, write outputs as CSV (comma-separated values) '
                  'files when possible.')

FLAGS = flags.FLAGS

MIC_SAMPLE_RATE_HZ = 15625.0
# Buffer duration in seconds.
BUFFER_DURATION_S = 64 / MIC_SAMPLE_RATE_HZ


def print_ports() -> None:
  """Prints a list of available ports."""
  available_ports = [tuple(p) for p in list(serial.tools.list_ports.comports())]
  if not available_ports:
    print('No available ports found.')
    print('Please make sure the device is connected and turned on.')
  else:
    print('Available ports:')
    for port in available_ports:
      print('%-20s %s' % port[:2])
    print('')


def run_tap_out(port: str,
                capture: List[str],
                duration_s: float
                ) -> Tuple[Dict[str, tap_out.Descriptor],
                           datetime.date,
                           Dict[str, tap_out.OutputData]]:
  """Connects to the device and runs the tap_out protocol.

  Args:
    port: String, name of the port where the device is connected.
    capture: List of strings, names of the tap_out outputs to capture.
    duration_s: Float, recording duration in seconds.
  Returns:
    (descriptors, build_date, captured) tuple, with the output descriptors,
    firmware build date, and captured output data.
  """
  print(f'Connecting to {port}...')

  try:
    with serial.Serial(port, baudrate=115200, timeout=0.1) as uart:
      comm = tap_out.TapOut(uart)

      descriptors, build_date = comm.get_descriptors()
      print(f'Firmware built on {build_date}')
      print('Available outputs:')
      for descriptor in descriptors.values():
        print(f'  {descriptor}')

      if capture:
        print('Recording...')
        num_buffers = int(np.ceil(duration_s / BUFFER_DURATION_S))
        captured = comm.capture(capture, num_buffers)
      else:
        captured = {}
  except KeyboardInterrupt:  # Exit gracefully on Ctrl+C.
    sys.exit(1)

  print('Done.\n')
  return descriptors, build_date, captured


def save_output(output_dir: str, start_time: datetime.datetime,
                descriptors: Dict[str, tap_out.Descriptor],
                build_date: datetime.date,
                captured: Dict[str, tap_out.OutputData]) -> None:
  """Saves output .npz, plots, and report to `output_dir`.

  Args:
    output_dir: String, output directory for saving .npz and report.
    start_time: Datetime, the time the program started.
    descriptors: Dict of Descriptors read by tap_out.
    build_date: Date when the device firmware was built.
    captured: Dict of captured tap_out output data.
  """
  os.makedirs(output_dir, exist_ok=True)
  save_npz(output_dir, captured)

  title = os.path.basename(output_dir)
  report = plot.HtmlReport(
      os.path.join(output_dir, 'report.html'), title)
  report.write(f'<p>Recorded at {start_time.strftime("%Y-%m-%d %H:%M:%S")}</p>')
  report.write(f'<p>Firmware built on {build_date}</p>')

  if 'mic_input' in captured:
    # If mic_input was captured, add audio playback widget to report page.
    report.write("""
<div><span style="vertical-align:middle">mic_input:</span>
<audio controls style="vertical-align:middle"><source src="mic_input.wav"
type="audio/wav"></audio></div>""")

  for name, data in captured.items():
    if not isinstance(data, np.ndarray):
      continue

    if name == 'mic_input':
      dsp.write_wav_file(
          os.path.join(output_dir, 'mic_input.wav'), data,
          int(round(MIC_SAMPLE_RATE_HZ)))
    elif name == 'tactile_output':
      samples = (2.0 / 255) * data.astype(float) - 1.0
      samples = np.round(32767 * samples.clip(-1.0, 1.0)).astype(np.int16)
      sample_rate_hz = int(round(get_sample_rate(descriptors[name])))
      dsp.write_wav_file(os.path.join(output_dir, 'tactile_output.wav'),
                         samples, sample_rate_hz)
    elif FLAGS.csv:
      save_csv(output_dir, name, data)

    fig = plot_capture(name, descriptors[name], data)
    if fig is not None:
      report.save_figure(os.path.join(output_dir, name + '.png'), fig)

  report.close()
  report_uri = pathlib.Path(os.path.abspath(report.filename)).as_uri()
  print(f'Output written to\n{report_uri}')


def get_sample_rate(descriptor: tap_out.Descriptor) -> float:
  """Gets the sample rate of a given tap_out output."""
  return descriptor.shape[0] / BUFFER_DURATION_S


def save_npz(output_dir: str, captured: Dict[str, tap_out.OutputData]) -> None:
  """Saves captured data as a numpy .npz file."""
  kw = {k: np.asarray(v) for k, v in captured.items()}
  np.savez(os.path.join(output_dir, 'captured.npz'), **kw)


def save_csv(output_dir: str, name: str, data: np.ndarray) -> None:
  """Saves a 1D or 2D output as a .csv file."""
  if 1 <= data.ndim <= 2:
    fmt = '%d' if np.issubdtype(data.dtype, np.integer) else '%.9g'
    np.savetxt(os.path.join(output_dir, name + '.csv'), data, fmt, ',')


def compute_stft(
    x: np.ndarray,
    fs: float,
    window_duration_s: float = 0.025
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes STFT with 50% overlap Hamming window."""
  window_size = int(round(window_duration_s * fs))
  hop = max(1, window_size // 2)
  window = np.hamming(window_size + 1)[:-1]
  fft_size = 2**int(np.ceil(np.log2(window_size)))

  segments = np.column_stack([
      x[start:start + window_size]
      for start in range(0, len(x) - window_size + 1, hop)])
  segments *= window[:, np.newaxis]
  stft = np.fft.rfft(segments, fft_size, axis=0)

  f = np.arange(stft.shape[0]) * (fs / fft_size)
  t = np.arange(stft.shape[1]) * (hop / fs)
  return f, t, stft


def plot_mic_input(
    data: np.ndarray, sample_rate_hz: float) -> matplotlib.figure.Figure:
  """Plots captured mic_input data."""
  samples = data.astype(float) / 2**15

  fig = matplotlib.figure.Figure(figsize=(9, 6))
  ax = fig.add_subplot(211)
  f, t, stft = compute_stft(samples, sample_rate_hz)
  image = np.abs(stft)**0.3
  vmin, vmax = 0.0, np.percentile(image, 99.9)
  ax.imshow(image, origin='lower', aspect='auto', cmap='density',
            extent=(t[0], t[-1], f[0], f[-1]), vmin=vmin, vmax=vmax)
  ax.set_ylim(0, 6000)
  ax.set_title('mic_input')
  ax.set_ylabel('Frequency (Hz)')

  ax = fig.add_subplot(212)
  t = np.arange(len(samples)) / sample_rate_hz
  ax.plot(t, samples)
  ax.axhline(y=0, color='k')
  ax.set_xlim(t[0], t[-1])
  ax.set_xlabel('Time (s)')
  return fig


def plot_carl_features(
    data: np.ndarray, sample_rate_hz: float) -> matplotlib.figure.Figure:
  """Plots captured CARL frontend features."""
  num_samples = data.shape[0]
  fig = matplotlib.figure.Figure(figsize=(9, 6))
  ax = fig.add_subplot(111)
  ax.imshow(data.transpose(), origin='lower', aspect='auto', cmap='density',
            extent=(0, num_samples / sample_rate_hz, 0, data.shape[1]),
            vmin=0.0)
  ax.set_title('carl_features')
  ax.set_ylabel('Channel')
  ax.set_xlabel('Time (s)')
  return fig


def plot_vowel_coord(
    data: np.ndarray, sample_rate_hz: float) -> matplotlib.figure.Figure:
  """Plots captured vowel_coord data."""
  t = np.arange(data.shape[0]) / sample_rate_hz

  fig = matplotlib.figure.Figure(figsize=(9, 6))
  ax = fig.add_subplot(111)
  ax.plot(t, data[:, 0], '-', label='x')
  ax.plot(t, data[:, 1], '-', label='y')
  ax.legend()
  ax.set_title('vowel_coord')
  ax.set_xlim(t[0], t[-1])
  ax.set_ylim(-1.0, 1.0)
  ax.set_xlabel('Time (s)')
  ax.set_ylabel('Embedding value')
  return fig


def plot_enveloper_energies(name: str,
                            data: np.ndarray,
                            sample_rate_hz: float) -> matplotlib.figure.Figure:
  """Plots captured enveloper energy data."""
  channel_names = ('Baseband', 'Vowel', 'Sh fric.', 'Fricative')
  num_channels = data.shape[1]
  t = np.arange(data.shape[0]) / sample_rate_hz
  fig = matplotlib.figure.Figure(figsize=(9, 6))

  for c in range(num_channels):
    ax = fig.add_subplot(num_channels, 1, c + 1)
    ax.plot(t, 10 * np.log10(data[:, c]))
    ax.set_xlim(t[0], t[-1])
    if c == 0:
      ax.set_title(name)
    ax.set_ylabel(channel_names[c] + ' (dB)')

  ax.set_xlabel('Time (s)')
  return fig


def plot_tactile_output(
    data: np.ndarray, sample_rate_hz: float) -> matplotlib.figure.Figure:
  """Plots captured tactile_output data."""
  samples = (2.0 / 255) * data.astype(float) - 1.0
  num_channels = samples.shape[1]
  t = np.arange(samples.shape[0]) / sample_rate_hz

  fig = matplotlib.figure.Figure(figsize=(9, 10))
  for c in range(num_channels):
    ax = fig.add_subplot(num_channels, 1, c + 1)
    ax.plot(t, samples[:, c])
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-1.0, 1.0)
    if c == 0:
      ax.set_title('tactile_output')
    ax.set_ylabel(f'Tactor {c + 1}')

  ax.set_xlabel('Time (s)')
  return fig


def plot_capture(name: str,
                 descriptor: tap_out.Descriptor,
                 data: np.ndarray) -> Optional[matplotlib.figure.Figure]:
  """Plots captured data."""
  if descriptor.dtype == tap_out.DType.TEXT:
    return None

  num_samples = data.shape[0]
  sample_rate_hz = get_sample_rate(descriptor)

  if name == 'mic_input':
    return plot_mic_input(data, sample_rate_hz)
  elif name == 'carl_features':
    return plot_carl_features(data, sample_rate_hz)
  elif name == 'tactile_output':
    return plot_tactile_output(data, sample_rate_hz)
  elif name == 'vowel_coord':
    return plot_vowel_coord(data, sample_rate_hz)
  elif name == 'smoothed_energy' or name == 'noise_energy':
    return plot_enveloper_energies(name, data, sample_rate_hz)
  elif len(data.shape) == 1:
    fig = matplotlib.figure.Figure(figsize=(9, 6))
    t = np.arange(num_samples) / sample_rate_hz
    ax = fig.add_subplot(111)
    ax.plot(t, data)
    ax.set_title(name)
    ax.set_xlabel('Time (s)')
  elif len(data.shape) == 2:
    num_channels = data.shape[1]
    t = np.arange(num_samples) / sample_rate_hz
    fig = matplotlib.figure.Figure(figsize=(9, 6))

    if num_channels <= 8:
      for c in range(num_channels):
        ax = fig.add_subplot(num_channels, 1, c + 1)
        ax.plot(t, data[:, c])
        ax.set_xlim(t[0], t[-1])
        if c == 0:
          ax.set_title(name)
        ax.set_ylabel(f'{c}')

      ax.set_xlabel('Time (s)')
    else:
      ax = fig.add_subplot(111)
      ax.imshow(data.transpose(), origin='upper', aspect='auto', cmap='density',
                extent=(0, num_samples / sample_rate_hz, 0, data.shape[1]),
                vmin=0.0)
      ax.set_title(name)
      ax.set_xlabel('Time (s)')

    return fig


def main(_) -> int:
  if not FLAGS.port:
    print_ports()
    return 1

  start_time = datetime.datetime.now()
  output_dir = FLAGS.output
  if not output_dir:
    output_dir = os.path.expanduser(
        os.path.join('~', 'tap_out_' + start_time.strftime('%Y%m%d_%H%M%S')))

  descriptors, build_date, captured = run_tap_out(
      FLAGS.port, FLAGS.capture, FLAGS.duration)

  save_output(output_dir, start_time, descriptors, build_date, captured)
  return 0


if __name__ == '__main__':
  app.run(main)
