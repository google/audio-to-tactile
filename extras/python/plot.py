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

"""Python plotting utilities."""

import os.path
import textwrap
from typing import Any, IO, Iterable, Optional, Sequence, Tuple, Union

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm
import matplotlib.figure
import numpy as np


def _get_filename(f: Union[str, IO[Any]]) -> Optional[str]:
  if isinstance(f, str):
    return f
  elif hasattr(f, 'name'):
    return str(f.name)
  else:
    return None


class HtmlReport:
  """Class for generating HTML reports."""

  def __init__(self,
               filename: Union[str, IO[str]],
               title: str,
               include_in_header: str = ''):
    """Constructor.

    Args:
      filename: String filename or a file object for writing the report.
      title: String, the report title.
      include_in_header: String, code to include in the HTML header.
    """
    if isinstance(filename, str):
      self._file = open(filename, 'wt')
      self.filename = filename
    else:
      self._file = filename
      self.filename = _get_filename(self._file)

    self.write(self._HEADER_TEMPLATE.format(
        title=title, include_in_header=include_in_header, css=self._CSS))

  def close(self) -> None:
    """Closes the HTML report."""
    self._file.write('</body></html>')
    self._file.close()

  def write(self, html: str) -> None:
    """Writes HTML code into the document. Inserts a newline after `html`."""
    self._file.write(html + '\n')

  def write_table(self,
                  table: Iterable[Iterable[str]],
                  first_row_is_header: bool = False) -> None:
    """Writes an HTML table.

    Example:
      report.write_table([['A1', 'B1', 'C1'],  # Table with 2 rows, 3 columns.
                          ['A2', 'B2', 'C2']])

    Args:
      table: List of list of strings, or other nested iterable.
      first_row_is_header: Bool. If true, the first row is formatted as <th>
        header cells.
    """
    wrap = lambda tag, content: f'<{tag}>{content}</{tag}>'
    self.write('<table>')
    for i, row in enumerate(table):
      col_tag = 'th' if first_row_is_header and i == 0 else 'td'
      self.write(wrap('tr', ''.join(wrap(col_tag, s) for s in row)))
    self.write('</table>')

  def save_figure(self,
                  image_filename: Union[str, IO[bytes]],
                  fig: matplotlib.figure.Figure,
                  **kw) -> None:
    """Saves a matplotlib Figure and adds it into the report.

    Args:
      image_filename: String or file object for saving the figure image.
      fig: Matplotlib Figure.
      **kw: Additional arguments to pass to plot.save_figure().
    """
    width, height = save_figure(image_filename, fig, **kw)
    path = _get_filename(image_filename)
    if self.filename is not None:
      path = os.path.relpath(path, os.path.dirname(self.filename))
    self._file.write(f'<img src="{path}" width={width} height={height}>\n')

  _HEADER_TEMPLATE = textwrap.dedent("""
      <!DOCTYPE html>

      <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en"><head>
      <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
      <style type="text/css">
      {css}
      </style>
      {include_in_header}
      <title>{title}</title></head>
      <body>
      <h1>{title}</h1>
      """).strip()
  _CSS = textwrap.dedent("""
      table {
        margin-bottom: 2em;
        border-bottom: 1px solid #ddd;
        border-right: 1px solid #ddd;
        border-spacing: 0;
        border-collapse: collapse;
      }
      table th {
        padding: .2em 1em;
        background-color: #eee;
        border-top: 1px solid #ddd;
        border-left: 1px solid #ddd;
      }
      table td {
        padding: .2em 1em;
        border-top: 1px solid #ddd;
        border-left: 1px solid #ddd;
        vertical-align: top;
      }
      pre {
        background-color: #f6f6f6;
        border: 1px solid #ccc;
        padding-left: 0.5em;
      }
      """).strip()


def plot_matrix_figure(
    matrix: Iterable[Any],
    tick_labels: Sequence[str],
    figsize: Tuple[float, float] = (7, 5),
    title: Optional[str] = None,
    row_label: Optional[str] = None,
    col_label: Optional[str] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: Optional[str] = 'Blues') -> matplotlib.figure.Figure:
  """Plots a 2D matrix as an image with element values as text labels."""
  matrix = np.asarray(matrix)
  size = len(matrix)
  if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
    raise ValueError(f'matrix has invalid shape {matrix.shape}')
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
        ax.text(j, i, f'{value:.2}', ha='center', va='center',
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


def save_figure(filename: Union[str, IO[bytes]],
                fig: matplotlib.figure.Figure,
                quality: Optional[int] = 90,
                optimize: Optional[bool] = True) -> Tuple[int, int]:
  """Use Agg to save a figure to an image file and return dimensions."""
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  _, _, width, height = canvas.figure.bbox.bounds
  fig.savefig(filename, pil_kwargs={'quality': quality, 'optimize': optimize})
  return int(width), int(height)


def _rational_polynomial_colormap(
    name: str,
    coeffs: Iterable[Any]) -> matplotlib.colors.LinearSegmentedColormap:
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
