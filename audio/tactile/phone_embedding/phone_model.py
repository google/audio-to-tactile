# Lint as: python3
r"""Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.


Train and eval a network for mapping audio to 2D vowel space coordinate.
"""

import datetime
import functools
import itertools
import os
import os.path
import random
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from absl import flags
import dataclasses
import haiku as hk
import jax
from jax.experimental import optix
import jax.numpy as jnp
import matplotlib
import matplotlib.figure
import numpy as np
import scipy.ndimage

from audio.tactile.jax import hk_util
from audio.tactile.phone_embedding import phone_util
from audio.tactile.python import plot
from audio.tactile.python import stats

# By default, train to classify these monophthong vowel classes.
# Additionally, TIMIT has these consonant classes (after merging several very
# similar classes):
# r,z,n,f,dh,s,v,m,l,sh,hh,ng,w,q,y,k,th,el,p,ch,t,en,jh,g,b,zh,d,em,dx,nx,eng
# and there is a "sil" (silence) class for pauses between speech.
DEFAULT_CLASSES = 'aa,uw,ih,iy,eh,ae,ah,er'

FLAGS = flags.FLAGS

# Model hyperparameters.
flags.DEFINE_list('classes', DEFAULT_CLASSES,
                  'The model is trained to classify these phoneme classes.')
flags.DEFINE_list('hidden_units', ['16', '16'],
                  'List where the ith element represents the number of units '
                  'in the ith hidden layer.')
flags.DEFINE_float('h1_penalty', 1e-4,
                   'h1 regularizer penalty weight on the first layer.')
flags.DEFINE_float('l1_penalty', 1e-4,
                   'L1 regularizer penalty weight on the other layers.')
flags.DEFINE_float('disperse_penalty', 0.75,
                   'Penalty to disperse embedded points of different labels.')
flags.DEFINE_float('disperse_separation', 0.3,
                   'Parameter of disperse penalty.')
flags.DEFINE_float('mapping_penalty', 400.0,
                   'Penalty to encourage matching MAPPING_TARGETS.')
flags.DEFINE_float('mapping_delta', 0.1,
                   'Charbonnier delta parameter in mapping penalty.')

# Training flags.
flags.DEFINE_float('validation_fraction', 0.05,
                   'Fraction of training dataset to use for validation.')
flags.DEFINE_integer('num_epochs', 10,
                     'Number of training epochs.')
flags.DEFINE_integer('batch_size', 512,
                     'Number of training examples per batch.')


def angle2cart(angle_deg: float, mag: float = 1.0) -> Tuple[float, float]:
  theta = angle_deg * np.pi / 180.0
  return (mag * np.cos(theta), mag * np.sin(theta))


# 2D target mapping coordinates for each class.
MAPPING_TARGETS = {
    'aa': angle2cart(-90),
    'uw': angle2cart(-30),
    'ih': angle2cart(30),
    'iy': angle2cart(90),
    'eh': angle2cart(150),
    'ae': angle2cart(-150),
    'ah': (0.0, 0.0),
    'er': angle2cart(0),
    'uh': angle2cart(-90, 0.5),
}


@dataclasses.dataclass
class Metadata:
  """Metadata for phone model."""

  classes: Sequence[str]
  hidden_units: Sequence[int]
  h1_penalty: float
  l1_penalty: float
  disperse_penalty: float
  disperse_separation: float
  mapping_penalty: float
  mapping_delta: float
  validation_fraction: float
  num_epochs: int
  batch_size: int
  dataset_metadata: Optional[Mapping[str, Any]] = None

  @staticmethod
  def from_flags() -> 'Metadata':
    """Construct Metadata from flags."""
    return Metadata(
        classes=FLAGS.classes,
        hidden_units=tuple(int(units) for units in FLAGS.hidden_units),
        h1_penalty=FLAGS.h1_penalty,
        l1_penalty=FLAGS.l1_penalty,
        disperse_penalty=FLAGS.disperse_penalty,
        disperse_separation=FLAGS.disperse_separation,
        mapping_penalty=FLAGS.mapping_penalty,
        mapping_delta=FLAGS.mapping_delta,
        validation_fraction=FLAGS.validation_fraction,
        num_epochs=FLAGS.num_epochs,
        batch_size=FLAGS.batch_size,
    )


def load_dataset(npz_file: str,
                 classes: Sequence[str],
                 class_weights: Optional[Dict[str, float]] = None,
                 ) -> phone_util.Dataset:
  """Loads training or testing data from a numpy .npz file.

  The .npz file holds 3D arrays of examples. The arrays are named according to
  which class they represent, e.g. an array named 'ae' represents examples with
  ground truth label 'ae'.

  Args:
    npz_file: String, npz filename.
    classes: List of phoneme class names to train the model to classify.
    class_weights: Dict, class weights for randomly subsampling the data. The
      fraction of examples retained for class `phone` is
      `class_weights.get(phone, 1.0) / max(class_weights.value())`
  Returns:
    Dataset.
  """
  if class_weights is None:
    class_weights = {}
  max_weight = max(class_weights.values()) if class_weights else 1.0
  class_weights = {phone: class_weights.get(phone, 1.0) / max_weight
                   for phone in classes}

  dataset = phone_util.read_dataset_npz(npz_file)
  dataset.subsample(class_weights)
  return dataset


def embedding_regularizer(embedded: jnp.ndarray,
                          labels: jnp.ndarray,
                          meta: Metadata) -> jnp.ndarray:
  embedded = embedded[:, -1, :]
  batch_size = embedded.shape[0]

  # Penalize close points of different labels according to
  #   1 / (1 + (min(distance, separation) / separation)^2)
  # Comparing all pairs of points would cost quadratically with batch size. To
  # reduce the cost to linear, we compare only the first point to the rest of
  # the batch.
  p0 = embedded[0]
  others = embedded[1:]
  separation_sqr = meta.disperse_separation**2
  dist_sqr = (p0[0] - others[:, 0])**2 + (p0[1] - others[:, 1])**2
  penalties = meta.disperse_penalty * jnp.sum(
      jnp.where(labels[0] != labels[1:], 1, 0.0) *
      1 / (1 + jnp.minimum(dist_sqr, separation_sqr) / separation_sqr))

  # For phones in MAPPING_TARGETS, use a Charbonnier loss to encourage points of
  # those labels to be close to the target.
  for i, phone in enumerate(meta.classes):
    if phone in MAPPING_TARGETS:
      tx, ty = MAPPING_TARGETS[phone]
      penalties += meta.mapping_penalty * hk_util.charbonnier_loss_on_squared(
          jnp.dot(labels == i,
            (embedded[:, 0] - tx)**2 + (embedded[:, 1] - ty)**2),
          meta.mapping_delta)

  return penalties / batch_size


def model_fun(batch, meta: Metadata) -> Dict[str, jnp.ndarray]:
  """Builds model for phone mapping."""
  hidden_units = meta.hidden_units
  num_frames = 1 + meta.dataset_metadata['num_frames_left_context']
  num_channels = meta.dataset_metadata['num_channels']

  penalties = 0.0
  # The network input x has shape (batch, num_frames, num_channels), where
  # typically batch=512, num_frames=3, num_channels=56.
  x = batch['observed'].astype(jnp.float32)

  # Compute mean PCEN power of the frame.
  mean_power = jnp.mean(x, axis=-1, keepdims=True)

  #### Encoder. ####
  # The first few layers of the network process each frame independently. We
  # temporarily reshape to (batch * num_frames, num_channels), flattening the
  # frames dimension into the batch dimension.
  x = jnp.reshape(x, (-1, num_channels))

  h1_regularizer = lambda w: FLAGS.h1_penalty * hk_util.h1_loss(w)
  l1_regularizer = lambda w: FLAGS.l1_penalty * hk_util.l1_loss(w)
  # Apply several fully-connected layers. Use H1 regularization on the first
  # layer to encourage smoothness along the channel dimension.
  for i, units in enumerate(meta.hidden_units):
    w_regularizer = h1_regularizer if i == 0 else l1_regularizer
    x, penalty_term = hk_util.Linear(units, w_regularizer=w_regularizer)(x)
    penalties += penalty_term
    x = jax.nn.relu(x)

  # Bottleneck layer, mapping the frame down to a 2D embedding space. We use
  # tanh activation to restrict embedding to the square [-1, 1] x [-1, 1].
  x, penalty_term = hk_util.Linear(2, w_regularizer=None)(x)
  penalties += penalty_term

  # Constrain embedded point to the hexagon.
  embed_r = 1e-4 + hk_util.hexagon_norm(x[:, 0], x[:, 1])
  x *= (jax.lax.tanh(embed_r) / embed_r).reshape(-1, 1)

  # Now we reshape the frame dimension back out of the batch dimension. The next
  # steps will process the embedded frames jointly.
  embedded = x = jnp.reshape(x, (-1, num_frames, x.shape[-1]))

  # Concatenate with mean_power to make a 3D embedding space. This extra
  # dimension is meant as a proxy for the information in the energy envelope.
  x = jnp.concatenate((x, mean_power), axis=-1)

  #### Decoder. ####
  # Decoder with a fixed architecture of 16-unit hidden layer.
  x = hk.Flatten()(x)
  x, penalty_term = hk_util.Linear(16, w_regularizer=l1_regularizer)(x)
  penalties += penalty_term
  x = jax.nn.relu(x)

  # Final layer producing a score for each phone class.
  scores, penalty_term = hk_util.Linear(len(meta.classes),
                                        w_regularizer=l1_regularizer)(x)
  penalties += penalty_term

  return {'embedded': embedded,
          'scores': scores,
          'penalties': penalties}


def train_model(meta: Metadata,
                dataset: phone_util.Dataset) -> hk_util.TrainedModel:
  """Train the model."""
  model = hk.transform(functools.partial(model_fun, meta=meta))

  # Split off a separate validation dataset.
  dataset_val, dataset_train = dataset.split(meta.validation_fraction)

  def generate_batches(dataset: phone_util.Dataset, batch_size: int):
    """Partition into batches. Examples in any partial batch are dropped."""
    x, y = dataset.get_xy_arrays(meta.classes, shuffle=True)

    batch_size = min(batch_size, len(x))
    num_batches = len(x) // batch_size
    batches_x = x[:num_batches * batch_size].reshape(
        num_batches, batch_size, *x.shape[1:])
    batches_y = y[:num_batches * batch_size].reshape(
        num_batches, batch_size)
    return batches_x, batches_y

  train_x, train_y = generate_batches(dataset_train, batch_size=meta.batch_size)
  t_eval_x, t_eval_y = generate_batches(dataset_train, batch_size=10000)
  t_eval_batch = {'observed': t_eval_x[0], 'label': t_eval_y[0]}
  v_eval_x, v_eval_y = generate_batches(dataset_val, batch_size=10000)
  v_eval_batch = {'observed': v_eval_x[0], 'label': v_eval_y[0]}

  # Initialize network and optimizer.
  seed = np.uint64(random.getrandbits(64))
  params = model.init(jax.random.PRNGKey(seed),
      {'observed': train_x[0], 'label': train_y[0]})
  optimizer = optix.adam(1e-3)
  opt_state = optimizer.init(params)

  # Print model summary.
  print(hk_util.summarize_model(params))

  def loss_fun(params, batch):
    """Training loss to optimize."""
    outputs = model.apply(params, batch)
    labels = hk.one_hot(batch['label'], len(meta.classes))
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(outputs['scores']))
    softmax_xent /= labels.shape[0]
    disperse = embedding_regularizer(outputs['embedded'], batch['label'], meta)
    return softmax_xent + disperse + outputs['penalties']

  @jax.jit
  def train_step(params, opt_state, batch):
    """Learning update rule."""
    grads = jax.grad(loss_fun)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optix.apply_updates(params, updates)
    return new_params, opt_state

  @jax.jit
  def accuracy(params, batch):
    """Evaluate classification accuracy."""
    scores = model.apply(params, batch)['scores']
    return jnp.mean(jnp.argmax(scores, axis=-1) == batch['label'])

  # Training loop.
  num_steps = len(train_x) * meta.num_epochs
  step_digits = len(str(num_steps))
  step = 0
  for epoch in range(meta.num_epochs):
    for batch_x, batch_y in zip(train_x, train_y):
      step += 1
      train_batch = {'observed': batch_x, 'label': batch_y}
      final_step = (step == num_steps)
      if final_step or step % 500 == 0:
        # Periodically evaluate classification accuracy on train & test sets.
        train_accuracy = accuracy(params, t_eval_batch)
        val_accuracy = accuracy(params, v_eval_batch)
        train_accuracy, val_accuracy = jax.device_get(
            (train_accuracy, val_accuracy))
        #  print(f'[{step:-{step_digits}d}/{meta.num_steps}] train acc = '
              #  f'{train_accuracy:.4f}, val acc = {val_accuracy:.4f}')
        print(f'[{step:-{step_digits}d}/{num_steps}] train acc = '
              f'{train_accuracy:.4f}, val acc = {val_accuracy:.4f}')

      params, opt_state = train_step(params, opt_state, train_batch)

  return hk_util.TrainedModel(model, meta=meta, params=params)


def compute_2d_hists(labels: np.ndarray,
                     coords: np.ndarray,
                     num_classes: int,
                     num_bins: int = 50,
                     hist_smoothing_stddev=0.05) -> np.ndarray:
  """Compute a 2D histogram over [-1, 1] x [-1, 1] for each class."""
  bin_width = 2 / num_bins
  hist = np.empty((num_classes, num_bins, num_bins))

  for i in range(num_classes):
    mask = (labels == i)
    coords_i = coords[mask].reshape(-1, 2)
    x, y = coords_i[:, 0], coords_i[:, 1]
    hist[i] = np.histogram2d(y, x, bins=num_bins, range=([-1, 1], [-1, 1]))[0]

    # For a more reliable density estimate, smooth the histogram with a Gaussian
    # kernel with stddev `hist_smoothing_stddev`.
    sigma = hist_smoothing_stddev / bin_width
    hist[i] = scipy.ndimage.gaussian_filter(hist[i], sigma)
    # Normalize density to integrate to one.
    hist[i] /= 1e-12 + bin_width**2 * hist[i].sum()

  return hist


def get_subplot_shape(num_subplots: int) -> Tuple[int, int]:
  subplot_rows = max(1, int(np.sqrt(num_subplots)))
  subplot_cols = -(-num_subplots // subplot_rows)
  return subplot_rows, subplot_cols


def plot_spatial_hists(labels: np.ndarray,
                       logits: np.ndarray,
                       classes: Sequence[str]) -> matplotlib.figure.Figure:
  """Plot histograms of how each phoneme class maps spatially to tactors."""
  # Get the classes that are in both `classes` and MAPPING_TARGETS.
  targeted_classes = [phone for phone in classes if phone in MAPPING_TARGETS]
  logits = np.compress([phone in targeted_classes for phone in classes],
                       logits, axis=1)
  targets = np.vstack([MAPPING_TARGETS[phone] for phone in targeted_classes])

  # Map logits down to 2D coordinates by weighted average.
  softmax_logits = np.exp2(4.0 * logits)
  softmax_logits /= np.sum(softmax_logits, axis=1, keepdims=True)
  coords = np.dot(softmax_logits, targets)

  hist = compute_2d_hists(labels, coords, len(classes))

  # Line segment data for plotting Voronoi cell boundaries.
  voronoi_x = [[0.2454, 0.4907, 0.2454, -0.2454, -0.4907, -0.2454, 0.2454,
                0.4907, 0.2454, -0.2454, -0.4907, -0.2454],
               [0.4907, 0.2454, -0.2454, -0.4907, -0.2454, 0.2454, 0.5774, 1.0,
                0.5774, -0.5774, -1.0, -0.5774]]
  voronoi_y = [[-0.425, 0.0, 0.425, 0.425, 0.0, -0.425, -0.425, 0.0, 0.425,
                0.425, 0.0, -0.425],
               [0.0, 0.425, 0.425, 0.0, -0.425, -0.425, -1.0, 0.0, 1.0, 1.0,
                0.0, -1.0]]
  # For more plot contrast, allow a small number of pixels to clip.
  vmax = np.percentile(hist, 99.7)

  fig = matplotlib.figure.Figure(figsize=(9, 6))
  subplot_rows, subplot_cols = get_subplot_shape(len(classes))
  for i in range(len(classes)):
    ax = fig.add_subplot(subplot_rows, subplot_cols, i + 1)
    ax.imshow(hist[i], origin='lower', aspect='equal', interpolation='bicubic',
              cmap='density', vmin=0.0, vmax=vmax, extent=(-1, 1, -1, 1))
    ax.plot(voronoi_x, voronoi_y, 'w-', linewidth=0.7, alpha=0.9)
    ax.plot(0.98 * targets[:, 0], 0.98 * targets[:, 1], 'ko', alpha=0.3)
    ax.set_title(classes[i], fontsize=14)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

  fig.suptitle('Spatial histograms', fontsize=15)
  return fig


def plot_embedded_hists(
    labels: np.ndarray,
    embedded: np.ndarray,
    classes: Sequence[str],
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]:
  """Plot histograms of how each phoneme class maps in the embedding space."""
  hist = compute_2d_hists(labels, embedded, len(classes))

  # Make a figure that shows all classes together on the same plot.
  fig_merged = matplotlib.figure.Figure(figsize=(6, 6))
  ax = fig_merged.add_subplot(1, 1, 1)

  # Get a set of distinct colors by sampling from the 'rainbow' colormap.
  cmap = matplotlib.cm.get_cmap('rainbow')
  x = np.arange(len(classes)) / float(len(classes))
  colors = ['#%02X%02X%02X' % (r, g, b)
            for r, g, b in (220 * cmap(x)[:, :3]).astype(int)]

  for i in range(len(classes)):
    level_thresholds = np.max(hist[i]) * np.array([0.5, 1])
    kwargs = {'colors': colors[i], 'origin': 'lower',
              'extent': (-1, 1, -1, 1)}
    contour = ax.contour(hist[i], levels=level_thresholds, **kwargs)
    ax.clabel(contour, fmt=classes[i], colors=colors[i])
    ax.contourf(hist[i], levels=level_thresholds, alpha=0.2, **kwargs)

  ax.axhline(0.0, color='k', linewidth=0.7)
  ax.axvline(0.0, color='k', linewidth=0.7)
  ax.set_aspect('equal', 'datalim')
  ax.set_xlim(-1, 1)
  ax.set_ylim(-1, 1)
  ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
  ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

  fig_merged.suptitle('Embedding histograms merged', fontsize=15)

  # Make another figure with a separate plot for each class.
  # For more plot contrast, allow a small number of pixels to clip.
  vmax = np.percentile(hist, 99.7)

  fig_separate = matplotlib.figure.Figure(figsize=(9, 6))
  subplot_rows, subplot_cols = get_subplot_shape(len(classes))
  for i in range(len(classes)):
    ax = fig_separate.add_subplot(subplot_rows, subplot_cols, i + 1)
    ax.imshow(hist[i], origin='lower', aspect='equal', interpolation='bicubic',
              cmap='density', vmin=0.0, vmax=vmax, extent=(-1, 1, -1, 1))
    ax.axhline(0.0, color='w', linewidth=0.7, alpha=0.9)
    ax.axvline(0.0, color='w', linewidth=0.7, alpha=0.9)
    ax.set_title(classes[i], fontsize=14)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

  fig_separate.suptitle('Embedding histograms', fontsize=15)

  return fig_merged, fig_separate


def plot_kernels(params: hk.Params,
                 layer_name: str,
                 max_num: int = 9) -> matplotlib.figure.Figure:
  """Plot kernels from layer `layer_name`.

  Args:
    params: Model params dict.
    layer_name: String.
    max_num: Integer, max number of kernels to plot. The kernels with the most
      energy are plotted.
  Returns:
    Matplotlib figure.
  """
  kernel = np.asarray(params[layer_name]['w'])
  top_index = np.argsort(np.sum(kernel**2, axis=0))[::-1][:max_num]
  num_taps = kernel.shape[0]

  fig = matplotlib.figure.Figure(figsize=(9, 6))
  subplot_rows, subplot_cols = get_subplot_shape(len(top_index))
  for i in range(len(top_index)):
    ax = fig.add_subplot(subplot_rows, subplot_cols, i + 1)
    ax.plot(kernel[:, top_index[i]], '.-' if num_taps < 60 else '-')
    ax.axhline(y=0, color='k')

  fig.suptitle(f'{layer_name} kernels', fontsize=15)
  return fig


def eval_model(model: hk_util.TrainedModel,
               dataset: phone_util.Dataset,
               output_dir: str) -> None:
  """Evaluate model and write HTML report to output directory."""
  classes = model.meta.classes
  x_test, y_test = dataset.get_xy_arrays(classes)
  outputs = model({'observed': x_test})
  scores = np.asarray(outputs['scores'])

  s = stats.MulticlassClassifierStats(len(classes))
  s.accum(y_test, scores)
  d_primes = s.d_prime
  confusion = s.confusion_matrix.astype(np.float32)

  # Compute normalized confusion matrix.
  confusion /= 1e-12 + confusion.sum(axis=1, keepdims=True)
  information_transfer = stats.estimate_information_transfer(confusion)
  mean_per_class_accuracy = np.mean(np.diag(confusion))

  print('mean d-prime: %.4f' % d_primes.mean())
  print('information transfer: %.2f' % information_transfer)
  print('mean per class accuracy: %.4f' % mean_per_class_accuracy)

  # Write HTML report.
  def output_file(*args):
    return os.path.join(output_dir, *args)

  os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
  report = plot.HtmlReport(output_file('report.html'), 'Eval')
  report.write('<p>training completed: %s</p>'
               % datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
  report.write('<p>mean d-prime: %.4f</p>' % d_primes.mean())
  report.write('<p>information transfer: %.2f</p>'
               % information_transfer)
  report.write('<p>mean per class accuracy: %.4f</p>'
               % mean_per_class_accuracy)

  report.write('<pre>')
  report.write(hk_util.summarize_model(model.params))
  report.write('</pre>')

  report.write('<table><tr><th>phone</th><th>d-prime</th></tr>')
  for phone, d_prime in zip(classes, d_primes):
    report.write(f'<tr><td>{phone}</td><td>{d_prime:.4f}</td></tr>')
  report.write('</table>')

  # Plot confusion matrix.
  fig = plot.plot_matrix_figure(
      confusion, classes, title='Normalized confusion matrix',
      row_label='True phone', col_label='Predicted phone')
  report.save_figure(output_file('images', 'confusion.png'), fig)
  del fig

  # Plot histograms of how each phoneme class maps spatially to tactors.
  fig = plot_spatial_hists(y_test, scores, classes)
  report.save_figure(output_file('images', 'spatial_hists.png'), fig)
  del fig

  if 'embedded' in outputs:
    embedded = np.asarray(outputs['embedded'])
    fig_merged, fig_separate = plot_embedded_hists(y_test, embedded, classes)
    report.save_figure(output_file('images', 'embedded_hists_merged.png'),
                       fig_merged)
    report.save_figure(output_file('images', 'embedded_hists.png'),
                       fig_separate)
    del fig_merged
    del fig_separate

  # Plot kernels for each layer. This is useful for tuning regularization.
  layer_names = sorted(model.params.keys())
  for i, layer_name in enumerate(layer_names):
    if layer_name.startswith('linear'):
      fig = plot_kernels(model.params, layer_name)
      report.save_figure(output_file('images', 'kernels%d.png' % i), fig)
      del fig

  report.close()
  print('\nfile://' + os.path.abspath(os.path.join(output_dir, 'report.html')))

