# Copyright 2019 Google LLC
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

r"""Python demo of running TactileWorker on microphone or WAV file input.

NOTE: This program needs the TactileWorker Python bindings:
  make tactile_worker.so

This program runs audio to tactile processing on microphone or WAV input,
producing 10-channel tactile signal output. The intention is that these tactile
channels are rendered on the TAPS sleeve with tactors in the following
arrangement:

  1: baseband  6: eh                        (7)-(6)   sh fricative
  2: aa        7: ae                        /     \      (9)
  3: uw        8: uh               (1)    (2) (8) (5)
  4: ih        9: sh fricative   baseband   \     /     (10)
  5: iy       10: fricative                 (3)-(4)   fricative
                                         vowel cluster

Use --channels to map tactile signals to output channels. For instance,
--channels=3,1,2,2 plays signal 3 on channel 1, signal 1 on channel 2, and
signal 2 on channels 3 and 4. A "0" in the channels list means that channel
is filled with zeros, e.g. --channels=1,0,2 sets channel 2 to zeros.

Flags:
 --input=<name>             Input device to get source audio from.
 --input=<wavfile>          Alternatively, input can be read from a WAV file.
                            The WAV file determines the sample rate.
 --output=<name>            Output device to play tactor signals to.
 --sample_rate_hz=<int>     Sample rate. Note that most devices only support
                            a few standard audio sample rates, e.g. 44100.
 --channels=<list>          Channel mapping.
 --channel_gains_db=<list>  Gains in dB for each channel. Usually negative
                            values, to avoid clipping. More negative value
                            means more attenuation, for example -13 is lower
                            in level than -10.
 --chunk_size=<int>         Frames per PortAudio buffer. (Default 256).
 --cutoff_hz=<float>        Cutoff in Hz for energy smoothing filters.
"""

import argparse
import sys
import time

import numpy as np

from dsp.python import wav_io
from tactile.python import tactile_worker


def volume_meter(a):
  """Text art volume meter, where 0 <= a <= 1."""
  level = min(max(int(round(a * 40)), 0), 40)
  full_blocks = min(4, level // 8)
  bar = (u'\u2588' * full_blocks +
         u' \u258F\u258E\u258D\u258C\u258B\u258A\u2589\u2588'[level -
                                                              full_blocks * 8])
  bar += u' ' * (5 - len(bar))
  return u'[\x1b[1;32m%s\x1b[0m]' % bar


def main(argv):
  parser = argparse.ArgumentParser(description='TactileProcessor Python demo')
  parser.add_argument('--input', type=str, help='Input WAV or device')
  parser.add_argument('--output', type=str, help='Output device')
  parser.add_argument('--sample_rate_hz', default=16000, type=int)
  parser.add_argument('--channels', type=str)
  parser.add_argument('--channel_gains_db', type=str, default='')
  parser.add_argument('--chunk_size', default=256, type=int)
  parser.add_argument('--cutoff_hz', default=500.0, type=float)
  parser.add_argument('--global_gain_db', default=0.0, type=float)
  parser.add_argument('--use_equalizer',
                      dest='use_equalizer', action='store_true')
  parser.add_argument('--nouse_equalizer',
                      dest='use_equalizer', action='store_false')
  parser.set_defaults(use_equalizer=True)
  parser.add_argument('--mid_gain_db', default=-10.0, type=float)
  parser.add_argument('--high_gain_db', default=-5.5, type=float)
  args = parser.parse_args(argv[1:])

  for arg_name in ('input', 'output', 'channels'):
    if not getattr(args, arg_name):
      print('Must specify --%s' % arg_name)
      return

  play_wav_file = args.input.endswith('.wav')

  if play_wav_file:
    wav_samples, sample_rate_hz = wav_io.read_wav_file(
        args.input, dtype=np.float32)
    wav_samples = wav_samples.mean(axis=1)
    input_device = None
  else:
    input_device = args.input
    sample_rate_hz = args.sample_rate_hz

  worker = tactile_worker.TactileWorker(
      input_device=input_device,
      output_device=args.output,
      sample_rate_hz=sample_rate_hz,
      channels=args.channels,
      channel_gains_db=args.channel_gains_db,
      global_gain_db=args.global_gain_db,
      chunk_size=args.chunk_size,
      cutoff_hz=args.cutoff_hz,
      use_equalizer=args.use_equalizer,
      mid_gain_db=args.mid_gain_db,
      high_gain_db=args.high_gain_db)

  if play_wav_file:
    worker.set_playback_input()
    worker.play(wav_samples)
  else:
    worker.set_mic_input()

  print('\nPress Ctrl+C to stop program.\n')
  print(' '.join('%-7s' % s for s in ('base', 'aa', 'uw', 'ih', 'iy', 'eh',
                                      'ae', 'uh', 'sh', 's')))
  rms_min, rms_max = 0.003, 0.05

  try:
    while True:
      # When playback is almost done, add wav_samples to the queue again. This
      # makes the WAV play in a loop.
      if (play_wav_file and
          worker.remaining_playback_samples < 0.1 * sample_rate_hz):
        worker.reset()
        worker.play(wav_samples)

      # Get the volume meters for each tactor and make a simple visualization.
      volume = worker.volume_meters
      activation = np.log2(1e-9 + volume / rms_min) / np.log2(rms_max / rms_min)
      activation = activation.clip(0.0, 1.0)
      print('\r' + ' '.join(volume_meter(a) for a in activation), end='')

      time.sleep(0.025)
  except KeyboardInterrupt:  # Stop gracefully on Ctrl+C.
    print('\n')


if __name__ == '__main__':
  main(sys.argv)
