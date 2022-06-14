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

"""Tests for plot.py."""

import io
import textwrap
import unittest

import matplotlib.cm
import matplotlib.figure
import numpy as np
from PIL import Image

from extras.python import plot


class PlotTest(unittest.TestCase):

  def test_html_report_write_table(self):
    """Test HtmlReport.write_table()."""
    fp = io.StringIO()  # Write report to in-memory file object.
    report = plot.HtmlReport(fp, 'Test report')
    self.assertIsNone(report.filename)
    report.write_table((('HA', 'HB'),
                        ('A1', 'B1'),
                        ('A2', 'B2'),
                        ('A3', 'B3')),
                       first_row_is_header=True)
    html = fp.getvalue()

    self.assertIn(
        textwrap.dedent("""
            <table>
            <tr><th>HA</th><th>HB</th></tr>
            <tr><td>A1</td><td>B1</td></tr>
            <tr><td>A2</td><td>B2</td></tr>
            <tr><td>A3</td><td>B3</td></tr>
            </table>
            """).strip(),
        html)

  def test_html_report_save_figure(self):
    """Test that save_figure() writes a readable image of expected size."""
    report_fp = io.StringIO()  # Write report to in-memory file object.
    report_fp.name = '/test/report/index.html'
    report = plot.HtmlReport(report_fp, 'Test report')
    self.assertEqual(report.filename, '/test/report/index.html')
    fig = matplotlib.figure.Figure(figsize=(7, 4.5), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    t = np.linspace(-12, 12, 200)
    ax.plot(t, np.sinc(t))

    fig_fp = io.BytesIO()
    fig_fp.name = '/test/report/images/fig.png'
    report.save_figure(fig_fp, fig)

    html = report_fp.getvalue()

    self.assertIn('<img src="images/fig.png" width=700 height=450>', html)
    self.assertTupleEqual((700, 450), Image.open(fig_fp).size)

  def test_density_colormap(self):
    """Test that 'density' colormap is registered with matplotlib."""
    cmap = matplotlib.cm.get_cmap('density')
    np.testing.assert_allclose(cmap(0.0), [0.214, 0.152, 0.535, 1], atol=0.001)
    np.testing.assert_allclose(cmap(1.0), [0.988, 0.978, 0.042, 1], atol=0.001)


if __name__ == '__main__':
  unittest.main()
