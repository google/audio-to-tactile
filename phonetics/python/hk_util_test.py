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

"""Test for hk_util."""

import functools
from typing import Sequence

from absl.testing import absltest
import dataclasses
import haiku as hk
import jax
from jax.experimental import optix
import jax.numpy as jnp
import numpy as np

from phonetics.python import hk_util


@dataclasses.dataclass
class FooMetadata:
  hidden_units: Sequence[int]


def foo_model(batch, meta: FooMetadata):
  """A dummy MLP model used for testing below."""
  x = batch['x']
  for i, units in enumerate(meta.hidden_units):
    x = hk.Linear(units)(x)
    if i < len(meta.hidden_units) - 1:
      x = jax.nn.relu(x)
  return x


class HaikuUtilTest(absltest.TestCase):

  def _assert_tree_equal(self, tree_x, tree_y):
    """Assert that two pytrees are equal."""
    self.assertEqual(jax.tree_util.tree_structure(tree_x),
                     jax.tree_util.tree_structure(tree_y))
    for x, y in zip(jax.tree_util.tree_leaves(tree_x),
                    jax.tree_util.tree_leaves(tree_y)):
      np.testing.assert_array_equal(x, y)

  def test_model_workflow(self):
    meta = FooMetadata(hidden_units=[5, 2])
    model = hk.transform(functools.partial(foo_model, meta=meta))

    # Get some random param values.
    batch = {'x': jnp.array([[0.5, 1.0, -1.5]])}
    params = model.init(jax.random.PRNGKey(0), batch)

    # Associate params with the model to get a TrainedModel.
    trained_model = hk_util.TrainedModel(model, meta=meta, params=params)

    # Save and load the model.
    filename = '/tmp/hk_util_test/model.pkl'
    trained_model.save(filename)

    recovered = hk_util.TrainedModel.load(filename, foo_model, FooMetadata)

    # Check that meta, params, and model forward function are the same.
    self.assertEqual(recovered.meta, meta)
    self._assert_tree_equal(recovered.params, params)
    y = recovered(batch)
    expected_y = model.apply(params, batch)
    np.testing.assert_array_equal(y, expected_y)

  def test_params_as_list(self):
    p = {'a': {'apple': jnp.array([1]),
               'avocado': jnp.array([2])},
         'b': {'beasts': {'baryonyx': jnp.array([3]),
                          'boar': jnp.array([4]),
                          'bull': jnp.array([5])}},
         'c': jnp.array([6])}
    self.assertEqual(hk_util.params_as_list(p), [
        ('a.apple', jnp.array([1])),
        ('a.avocado', jnp.array([2])),
        ('b.beasts.baryonyx', jnp.array([3])),
        ('b.beasts.boar', jnp.array([4])),
        ('b.beasts.bull', jnp.array([5])),
        ('c', jnp.array([6]))])

  def test_summarize_model(self):

    def model_fun(x):
      """A model with two submodules."""

      class Alpha(hk.Module):  # Alpha submodule.

        def __call__(self, x):
          return hk.Sequential([
              hk.Conv2D(8, (3, 3)), jax.nn.relu,
              hk.MaxPool((1, 2, 2, 1), (1, 2, 2, 1), 'VALID'),
              hk.Flatten(),
              hk.Linear(3, with_bias=False)
          ])(x)

      class Beta(hk.Module):  # Beta submodule.

        def __call__(self, x):
          return hk.Sequential([hk.Flatten(), hk.Linear(3), jax.nn.relu])(x)

      return hk.Linear(1)(Alpha()(x) + Beta()(x))

    model = hk.transform(model_fun)
    x = np.random.randn(1, 12, 15, 1)
    params = model.init(jax.random.PRNGKey(0), x)

    summary = hk_util.summarize_model(params)
    self.assertEqual(
        summary, """
Variable         Shape            #
alpha/conv2_d.b  (8,)             8
alpha/conv2_d.w  (3, 3, 1, 8)    72
alpha/linear.w   (336, 3)      1008
beta/linear.b    (3,)             3
beta/linear.w    (180, 3)       540
linear.b         (1,)             1
linear.w         (3, 1)           3
Total                          1635
""".strip())

  def test_losses(self):
    """Test losses on a few different input shapes."""
    np.random.seed(0)
    for shape in ((1,), (5,), (5, 4), (3, 4, 2)):
      x = np.random.randn(*shape).astype(np.float32)
      np.testing.assert_allclose(hk_util.l1_loss(x),
                                 np.sum(np.abs(x)), atol=5e-6)
      np.testing.assert_allclose(hk_util.l2_loss(x),
                                 np.sum(x**2), atol=5e-6)

      d = 0.3
      expected = np.sum(d**2 * (np.sqrt(1.0 + (x / d)**2) - 1.0))
      np.testing.assert_allclose(hk_util.charbonnier_loss(x, d),
                                 expected, atol=5e-6)
      np.testing.assert_allclose(hk_util.charbonnier_loss_on_squared(x**2, d),
                                 expected, atol=5e-6)

      if x.size > 1:
        np.testing.assert_allclose(hk_util.h1_loss(x, axis=0),
                                   np.sum(np.diff(x, axis=0)**2), atol=5e-6)
        np.testing.assert_allclose(hk_util.h1_loss(x, axis=-1),
                                   np.sum(np.diff(x, axis=-1)**2), atol=5e-6)

  def test_hexagon_norm(self):
    """Test hk_util.hexagon_norm()."""
    np.random.seed(0)
    theta = np.arange(6) * np.pi / 3
    vertices = np.stack((np.sin(theta), -np.cos(theta)))  # Hexagon vertices.

    n = np.random.randint(6, size=100)
    w = np.random.rand(100)
    # Get random points on the hexagon's boundary.
    x, y = (1.0 - w) * vertices[:, n] + w * vertices[:, (n + 1) % 6]

    # hexagon_norm is equal to 1.0 on the hexagon boundary.
    np.testing.assert_allclose(hk_util.hexagon_norm(x, y),
                               np.ones(100), atol=1e-6)

    # Absolute homogeneity: scaling by `r` changes the norm by factor `r`.
    r = 2.0 * np.random.rand(100)
    np.testing.assert_allclose(hk_util.hexagon_norm(r * x, r * y),
                               r, atol=1e-6)

  def test_linear(self):
    """Test hk_util.Linear() regularized linear layer."""
    def model_fun(x):
      return hk_util.Linear(
          5,
          w_init=hk.initializers.TruncatedNormal(1.0),
          b_init=hk.initializers.TruncatedNormal(1.0),
          w_regularizer=functools.partial(hk_util.h1_loss, axis=1),
          b_regularizer=hk_util.l1_loss,
          name='foo')(x)

    model = hk.transform(model_fun)
    x = jnp.array([-0.3, 0.5, 4.0], np.float32)
    params = model.init(jax.random.PRNGKey(0), x)
    y, penalties = model.apply(params, x)

    # Check params structure.
    self.assertIn('foo', params)
    self.assertIn('w', params['foo'])
    self.assertIn('b', params['foo'])
    self.assertEqual(params['foo']['w'].shape, (3, 5))
    self.assertEqual(params['foo']['b'].shape, (5,))

    # Check `y` output value.
    expected_y = jnp.dot(x, params['foo']['w']) + params['foo']['b']
    np.testing.assert_allclose(y, expected_y, atol=1e-6)

    # Check penalties.
    expected_penalties = (hk_util.h1_loss(params['foo']['w'], axis=1)
                          + hk_util.l1_loss(params['foo']['b']))
    np.testing.assert_allclose(penalties, expected_penalties, atol=1e-6)

  def test_linear_no_bias(self):
    """Test hk_util.Linear() without bias term."""
    def model_fun(x):
      return hk_util.Linear(
          5,
          use_bias=False,
          w_init=hk.initializers.TruncatedNormal(1.0),
          w_regularizer=functools.partial(hk_util.h1_loss, axis=0),
          name='foo')(x)

    model = hk.transform(model_fun)
    x = jnp.array([-0.3, 0.5, 4.0], np.float32)
    params = model.init(jax.random.PRNGKey(0), x)
    y, penalties = model.apply(params, x)

    # Check params structure.
    self.assertIn('foo', params)
    self.assertIn('w', params['foo'])
    self.assertNotIn('b', params['foo'])
    self.assertEqual(params['foo']['w'].shape, (3, 5))

    # Check `y` output value.
    expected_y = jnp.dot(x, params['foo']['w'])
    np.testing.assert_allclose(y, expected_y, atol=1e-6)

    # Check penalties.
    expected_penalties = hk_util.h1_loss(params['foo']['w'], axis=0)
    np.testing.assert_allclose(penalties, expected_penalties, atol=1e-6)

  def test_regularized_training(self):
    """Test that adding regularization penalty to the training loss works."""
    np.random.seed(0)
    # Set up the problem of recovering w given x and
    #   y = x . w + noise
    # with the a priori assumption that w is sparse. There are fewer examples
    # than dimensions (x is a wide matrix), so the problem is underdetermined
    # without the sparsity assumption.
    num_examples, num_dim = 8, 10
    x = np.random.randn(num_examples, num_dim).astype(np.float32)
    true_w = np.zeros((num_dim, 2), np.float32)
    true_w[[2, 4, 6], 0] = [1.0, 2.0, 3.0]
    true_w[[3, 5], 1] = [4.0, 5.0]
    y = np.dot(x, true_w) + 1e-3 * np.random.randn(num_examples, 2)

    # Get the least squares estimate for w. It isn't very accurate.
    least_squares_w = np.linalg.lstsq(x, y, rcond=None)[0]
    least_squares_w_error = hk_util.l2_loss(least_squares_w - true_w)

    # Get a better estimate by solving the L1 regularized problem
    #  argmin_w ||x . w - y||_2^2 + c ||w||_1.
    w_regularizer = lambda w: 4.0 * hk_util.l1_loss(w)
    def model_fun(batch):
      x = batch['x']
      return hk_util.Linear(2, use_bias=False, w_regularizer=w_regularizer)(x)

    model = hk.transform(model_fun)

    def loss_fun(params, batch):
      """Training loss with L1 regularization penalty term."""
      y_predicted, penalties = model.apply(params, batch)
      return hk_util.l2_loss(y_predicted - batch['y']) + penalties

    batch = {'x': x, 'y': y}
    params = model.init(jax.random.PRNGKey(0), batch)
    optimizer = optix.chain(  # Gradient descent with decreasing learning rate.
        optix.trace(decay=0.0, nesterov=False),
        optix.scale_by_schedule(lambda i: -0.05 / jnp.sqrt(1 + i)))
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch):
      grads = jax.grad(loss_fun)(params, batch)
      updates, opt_state = optimizer.update(grads, opt_state)
      new_params = optix.apply_updates(params, updates)
      return new_params, opt_state

    for _ in range(1000):
      params, opt_state = train_step(params, opt_state, batch)

    l1_w = params['linear']['w']
    l1_w_error = hk_util.l2_loss(l1_w - true_w).item()

    # The L1-regularized estimate is much more accurate.
    self.assertGreater(least_squares_w_error, 4.0)
    self.assertLess(l1_w_error, 1.0)


if __name__ == '__main__':
  absltest.main()
