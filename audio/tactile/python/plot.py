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


Python plotting utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import typing
from typing import Any, Iterable, Text, Tuple  # pylint:disable=unused-import

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm
import matplotlib.figure
import numpy as np


def _get_filename(f):
  # type: (typing.IO) -> Text
  if hasattr(f, 'read'):
    if hasattr(f, 'name'):
      return f.name
    else:
      raise ValueError('Unable to determine name of file')
  else:
    return str(f)


class HtmlReport(object):
  """Class for generating HTML reports."""

  def __init__(self, filename, title, include_in_header=''):
    # type: (typing.TextIO, Text, Text) -> None
    if hasattr(filename, 'read'):
      self._file = filename
      self.filename = _get_filename(self._file)
    else:
      self._file = open(filename, 'wt')
      self.filename = filename

    self._file.write("""<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en"><head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
{include_in_header}
<title>{title}</title></head><body>
<h1>{title}</h1>
""".format(
    title=title, include_in_header=include_in_header))

  def close(self):
    # type: () -> None
    self._file.write('</body></html>')
    self._file.close()

  def write(self, html):
    # type: (Text) -> None
    self._file.write(html + '\n')

  def save_figure(self, image_filename, fig, **kw):
    # type: (typing.BinaryIO, matplotlib.figure.Figure, Any) -> None
    width, height = save_figure(image_filename, fig, **kw)
    path = os.path.relpath(_get_filename(image_filename),
                           os.path.dirname(self.filename))
    self._file.write(f'<img src="{path}" width={width} height={height}>\n')


def plot_matrix_figure(matrix, tick_labels, figsize=(7, 5), title=None,
                       row_label=None, col_label=None,
                       vmin=0.0, vmax=1.0, cmap='Blues'):
  """Plots a 2D matrix as an image with element values as text labels."""
  matrix = np.asarray(matrix)
  size = len(matrix)
  if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
    raise ValueError('matrix has invalid shape %s' % (matrix.shape,))
  if len(tick_labels) != size:
    raise ValueError('tick_labels mismatches matrix')
  if vmin is None:
    vmin = matrix.min()
  if vmax is None:
    vmax = matrix.max()

  fig = matplotlib.figure.Figure(figsize=figsize)
  ax = fig.add_subplot(1, 1, 1)
  image = ax.imshow(matrix, interpolation='nearest', cmap=cmap,
                    vmin=vmin, vmax=vmax)

  if len(matrix) <= 12:
    midpoint = 0.5 * (vmin + vmax)
    for i in range(size):
      for j in range(size):
        value = matrix[i, j]
        ax.text(j, i, '%.2f' % value, ha='center', va='center',
                color='white' if value > midpoint else 'black')

  ax.figure.colorbar(image, ax=ax)
  ax.set(xticks=np.arange(size),
         yticks=np.arange(size),
         xticklabels=tick_labels,
         yticklabels=tick_labels)
  if title:
    ax.set_title(title, fontsize=15)
  if row_label:
    ax.set_ylabel(row_label)
  if col_label:
    ax.set_xlabel(col_label)
  fig.set_tight_layout(True)

  return fig


def save_figure(filename, fig, quality=90, optimize=True):
  # type: (typing.BinaryIO, matplotlib.figure.Figure, int, bool) -> Tuple
  """Use Agg to save a figure to an image file and return dimensions."""
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  _, _, width, height = canvas.figure.bbox.bounds
  fig.savefig(filename, quality=quality, optimize=optimize)
  return int(width), int(height)


def _rational_polynomial_colormap(name, coeffs):
  # type: (Text, Iterable) -> matplotlib.colors.LinearSegmentedColormap
  """Colormap where each channel is a rational polynomial P(x) / Q(x)."""
  x = np.linspace(0.0, 1.0, 256)
  colors = np.column_stack([np.polyval(p, x) / np.polyval(q, x)
                            for p, q in coeffs])
  return matplotlib.colors.LinearSegmentedColormap.from_list(name, colors)


# A nice perceptually-uniform blue->green->orange colormap.
matplotlib.cm.register_cmap(
    name='density',
    cmap=_rational_polynomial_colormap('density', [
        ([0.74167, -0.76313, -0.53475, 0.21633, 0.51098, 0.01247, -0.48087,
          0.58662, -0.36805, 0.11332, -0.01564, 0.00078],
         [1, -1.94771, 0.13094, 2.02604, -1.65808, 0.53783, -0.0727, 0.00364]),
        ([2.85072, -0.34097, -1.33016, -3.07746, 2.42046, 0.17642],
         [1, 1.50548, -0.52946, -2.84493, 0.42675, 1.15712]),
        ([0.87389, -1.60646, 0.95659, -0.25982, 0.02827, -0.00266],
         [1, -2.54503, 1.78386, -0.54413, 0.06914, -0.00497])]))
