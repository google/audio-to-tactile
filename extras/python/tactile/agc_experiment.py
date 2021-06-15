import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

#  from extras.python import auto_gain_control
from extras.python import dsp
from extras.python.tactile import energy_envelope
from extras.python.tactile import post_processor

sample_rate_hz = 16000
wav_file = '/home/getreuer/projects/tactile/sandwich-16khz.wav'
wav_dir = '/home/getreuer/projects/tactile/'

def read_wav(s):
  wav_samples, wav_sample_rate_hz = dsp.read_wav_file(
      wav_dir + s + '.wav', dtype=np.float32)
  assert wav_sample_rate_hz == 16000
  return wav_samples.mean(axis=1)


def a2db(x):
  return 20*np.log10(np.maximum(1e-7, np.abs(x)))
def p2db(x):
  return 10*np.log10(np.maximum(1e-14, np.abs(x)))


def genv(x, fs, tau):
  p = np.exp(-1/ (tau * fs))
  return scipy.signal.lfilter([1-p],[1,-p], np.abs(x))


def run_env(x, decimation, params):
  out = []
  for p in params:
    d = {}
    y = energy_envelope.EnergyEnvelope(
      sample_rate_hz, decimation, **p).process_samples(x)#, debug_out=d)
    d['output'] = y
    out.append(d)

  return out


name = 'transition-bg'
# name = 'see-them-in-bg'
x = read_wav(name)
t = np.arange(len(x)) / sample_rate_hz
#  x[(t % 4) > 2] *= 3

std_params = [energy_envelope.BASEBAND_PARAMS,
              energy_envelope.VOWEL_PARAMS,
              energy_envelope.SH_FRICATIVE_PARAMS,
              energy_envelope.FRICATIVE_PARAMS]
params = [dict(p) for p in std_params]

#  alpha = 0.7
#  beta = 0.25

#  params[0]['denoise_thresh_factor'] = 12.0
#  params[0]['agc_strength'] = alpha
#  params[0]['noise_tau_s'] = 0.4
#  params[0]['compressor_exponent'] = beta

#  params[1]['denoise_thresh_factor'] = 8.0
#  params[1]['agc_strength'] = alpha
#  params[1]['noise_tau_s'] = 0.4
#  params[1]['compressor_exponent'] = beta

#  params[2]['denoise_thresh_factor'] = 8.0
#  params[2]['agc_strength'] = alpha
#  params[2]['noise_tau_s'] = 0.4
#  params[2]['compressor_exponent'] = beta

#  params[3]['denoise_thresh_factor'] = 8.0
#  params[3]['agc_strength'] = alpha
#  params[3]['noise_tau_s'] = 0.4
#  params[3]['compressor_exponent'] = beta

decimation = 8
y = run_env(x, decimation, params)


fig = plt.figure(figsize=(7, 8.5))
fontsize = 13.5
ax = fig.subplots(5, 1, sharex=True)
#  ax[0].specgram(x, Fs=sample_rate_hz, NFFT=128,
               #  noverlap=64, cmap='Blues', vmin=-90, vmax=-40)
#  ax[0].set_ylim(0, 6500)
ax[0].plot(t, x, 'r')
#  ax[0].set_ylim(-1, 1)
a = np.abs(x).max()
ax[0].set_ylim(-a, a)
#  ax[0].set_title(f'"{name}"')
ax[0].set_title('Baseline', fontsize=1.2*fontsize)
ax[0].set_ylabel('Input', fontsize=fontsize)

name = ('Baseband', 'Vowel', 'SH fricative', 'Fricative')

t = np.arange(len(y[0]['output'])) * decimation / sample_rate_hz
for i in range(4):
  if False:
    ax[i+1].plot(t, p2db(y[i]['smoothed_energy']), 'r', label='input dB')
    ax[i+1].plot(t, p2db(2**y[i]['log2_noise']), '#bcbd22', label='noise dB')
    ax[i+1].plot(t, a2db(genv(y[i]['output'], sample_rate_hz/decimation, 0.005)),
                 color='#1f77b4', label='output dB')
    ax[i+1].plot(t, p2db(y[i]['smoothed_energy']), label='smoothed_energy')
    ax[i+1].plot(t, p2db(2**y[i]['log2_noise']), label='log2_noise')
    ax[i+1].plot(t, p2db(y[i]['smoothed_gain']), label='smoothed_gain')
    ax[i+1].plot(t, a2db(genv(y[i]['output'], sample_rate_hz/decimation, 0.005)), label='output')
    #  ax[i+1].plot(t, genv(y[i]['output'], sample_rate_hz/decimation, 0.01), label='output')
    ax[i+1].set_ylim(-90, 5)
  else:
    ax[i+1].plot(t, a2db(genv(y[i]['output'], sample_rate_hz/decimation, 0.005)),
                 color='#1f77b4', label='output dB')
    #  ax[i+1].plot(t, y[i]['output'])
  #  ax[i+1].plot(t, 10*np.log10(y[i]))
  #  ax[i+1].plot(t, a2db(y[i]))

  ax[i+1].set_ylim(-75, 0)
  #  ax[i+1].set_ylim(0, 1)
  ax[i+1].set_ylabel(name[i], fontsize=fontsize)
  #  ax[i+1].axhline(0, color='k')

#  ax[1].legend()
ax[-1].set_xlim(t[0], t[-1])
ax[-1].set_xlabel('Time (s)', fontsize=fontsize)
#  ax[-1].set_xlim(np.array([0.2, 0.75]) - 0.02)

plt.show()
