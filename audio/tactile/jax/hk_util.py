# Lint as: python3
r"""Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.


Some building blocks for working with JAX/Haiku.
"""

import functools
import json
import os
import os.path
import pickle
from typing import Any, Callable, List, Mapping, Optional, Tuple, Type

import dataclasses
import haiku as hk
import jax.numpy as jnp
import numpy as np

# Type alias for a regularizer, like `l2_loss()` below.
Regularizer = Callable[[jnp.ndarray], jnp.ndarray]


@dataclasses.dataclass
class TrainedModel:
  """A complete trained model with parameters.

  Fields:
    model_object: Object obtained from `hk.transform()` with .apply method.
    meta: Metadata of any user-defined configurations, hyperparameters, etc.
      Must be a dataclass of JSON-serializable fields.
    params: hk.Params for the model.
  """
  model_object: hk.Transformed
  meta: Any
  params: hk.Params

  def __call__(self, inputs):
    """Evaluate the model forward function on `inputs`."""
    return self.model_object.apply(self.params, inputs)

  def save(self, filename: str) -> None:
    """Save TrainedModel to `filename`.

    params is saved as a pickle, and meta is saved as JSON.

    Args:
      filename: String, output filename for the params pickle. The meta JSON is
        saved to the same filename with extension '.json'. The output directory
        is automatically created.
    """
    model_dir = os.path.dirname(filename)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'meta.json'), 'wt') as f:
      json.dump(dataclasses.asdict(self.meta), f, indent=2)
    with open(filename, 'wb') as f:
      pickle.dump(self.params, f)

  @classmethod
  def load(cls,
           filename: str,
           model_fun: Callable[[Any, Any], Any],
           meta_dataclass: Type[Any]):
    """Load TrainedModel from `filename` as written by `save()`.

    Args:
      filename: String, filename of params pickle. meta is read from JSON having
        the same filename with extension '.json'.
      model_fun: Model forward function of the form `model_fun(batch, meta)`.
      meta_dataclass: Type, the dataclass for meta.
    Returns:
      TrainedModel.
    """
    model_dir = os.path.dirname(filename)
    with open(os.path.join(model_dir, 'meta.json'), 'rt') as f:
      meta = meta_dataclass(**json.load(f))
    with open(filename, 'rb') as f:
      params = pickle.load(f)
    model_object = hk.transform(functools.partial(model_fun, meta=meta))
    return cls(model_object, meta, params)


def params_as_list(params: hk.Params) -> List[Tuple[str, jnp.ndarray]]:
  """Represents hk.Params as a list of ('node.subnode...name', array) 2-tuples.

  Args:
    params: hk.Params.
  Returns:
    List of (name, array) 2-tuples.
  """
  out = []

  def _iterate_node(name, node):
    if isinstance(node, Mapping):
      for k, v in node.items():
        _iterate_node(name + '.' + k, v)
    else:
      out.append((name, node))

  for k, v in params.items():
    _iterate_node(k, v)

  return out


def summarize_model(params: hk.Params) -> str:
  """Makes a summary of number of parameters, names, and shapes in a model.

  Args:
    params: hk.Params.
  Returns:
    String summary description of the model.
  """
  rows = [('Variable', 'Shape', '#')]
  for name, array in params_as_list(params):
    rows.append((name, array.shape, array.size))

  total_params = sum(size for _, _, size in rows[1:])
  rows.append(('Total', '', total_params))  # Add final "Total" line.

  # Format `rows` as a 3-column table.
  max_len = lambda j: max(len(str(row[j])) for row in rows)
  widths = [max_len(j) for j in range(3)]  # Get width for each column.
  return '\n'.join(
      f'{name:{widths[0]}}  {str(shape):{widths[1]}}  {str(size):>{widths[2]}}'
      for name, shape, size in rows)


def l1_loss(w: jnp.ndarray) -> jnp.ndarray:
  """L1 loss, sum_n |w[n]|."""
  return jnp.sum(jnp.abs(w))


def l2_loss(w: jnp.ndarray) -> jnp.ndarray:
  """L2 loss, sum_n |w[n]|^2."""
  return jnp.sum(jnp.abs(w)**2)


def h1_loss(w: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
  """H1 loss (L2 loss on difference), sum_n |w[n+1] - w[n]|^2."""
  return l2_loss(jnp.diff(w, axis=axis))


def charbonnier_loss(w: jnp.ndarray, delta) -> jnp.ndarray:
  """Charbonnier (pseudo-Huber) loss, sum_n delta^2 (sqrt(1 + (w/delta)^2) - 1).

  Computes a smooth approximation of Huber loss
  [https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function].

  Args:
    w: Array.
    delta: Positive float. The loss acts like L2 for |w| << delta and like L1
      for |w| >> delta.
  Returns:
    Scalar loss value.
  """
  return charbonnier_loss_on_squared(w**2, delta)


def charbonnier_loss_on_squared(w_squared: jnp.ndarray, delta) -> jnp.ndarray:
  """Charbonnier loss where the argument w is already squared."""
  return jnp.sum(delta**2 * (jnp.sqrt(1.0 + w_squared / delta**2) - 1.0))


def hexagon_norm(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Computes a 2D norm or distance to the origin for the hexagon.

  The "unit ball" set {(x,y) : hexagon_norm(x,y) == 1.0} is the hexagon with
  vertices

    (sin(n pi/3), -cos(n pi/3)), n = 0, ..., 5.

  Args:
    x: Array.
    y: Array of the same size.
  Returns:
    Array of the same size.
  """
  x = jnp.abs(x) / np.cos(np.pi / 6)
  y = jnp.abs(y)
  return jnp.maximum(x, 0.5 * x + y)


class Linear(hk.Module):
  """Like hk.Linear, but with regularizer on the weights and/or bias.

  This layer should be used like

    y, penalties = Linear(output_size, w_regularizer=l2_loss)(x)

  where `y` is the layer output and `penalties` is a JAX scalar of the
  regularization penalties computed on the layer parameters. Penalties from all
  layers should be accumulated and added to the training loss:

    def loss_fun(params, batch):
      outputs, summed_penalties = model.apply(params, batch['inputs'])
      return metric(outputs, batch['target']) + summed_penalties
  """

  def __init__(self,
               output_size: int,
               use_bias: bool = True,
               w_init: Optional[hk.initializers.Initializer] = None,
               b_init: Optional[hk.initializers.Initializer] = None,
               w_regularizer: Optional[Regularizer] = None,
               b_regularizer: Optional[Regularizer] = None,
               name: Optional[str] = None) -> None:
    super(Linear, self).__init__(name=name)
    self.input_size = None
    self.output_size = int(output_size)
    self.use_bias = bool(use_bias)
    self.w_init = w_init
    self.b_init = b_init or jnp.zeros
    self.w_regularizer = w_regularizer or (lambda _: 0.0)
    self.b_regularizer = b_regularizer or (lambda _: 0.0)

  def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Layer forward function.

    Args:
      x: Inputs.
    Returns:
      (outputs, penalties) 2-tuple.
    """
    self.input_size = x.shape[-1]

    if self.w_init is None:
      w_init = hk.initializers.TruncatedNormal(
          1.0 / np.sqrt(self.input_size))
    else:
      w_init = self.w_init

    w = hk.get_parameter('w', shape=[self.input_size, self.output_size],
                         dtype=x.dtype, init=w_init)
    y = jnp.dot(x, w)
    penalty = self.w_regularizer(w)

    if self.use_bias:
      b = hk.get_parameter('b', shape=[self.output_size],
                           dtype=x.dtype, init=self.b_init)
      y += jnp.broadcast_to(b, y.shape)
      penalty += self.b_regularizer(b)

    return y, penalty
