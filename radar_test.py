"""Tests for radar.py."""

import numbers
import os
import radar
import shutil
import tempfile
import unittest
from sklearn import tree

def make_tree():
  """Make a DecisionTreeClassifier."""
  pass  # TODO

class RadarTest(unittest.TestCase):
    """"Test for radar module."""

    def setUp(self):
        # Explicit msgs in asserts is appended to (rather than overriding) the
        # default failure messages.
        self.longMessage = True # pylint: disable=invalid-name
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_tree_data_format(self):
        data = radar.tree_data(make_tree())
        # There's an inner and outer ring.
        self.assertEqual(len(data), 2)
        for i, ring in enumerate(data):
            for j, item in enumerate(ring):
                msg = '\nFor data[%d][%d] = %r\n' % (i, j, item)
                self.assertItemsEqual(item.keys(), ['axis', 'value'], msg=msg)
                self.assertRegexpMatches(item['axis'], 'CV for .*', msg=msg)
                self.assertIsInstance(item['value'], numbers.Number, msg=msg)
                self.assertGreaterEqual(item['value'], 0.0, msg=msg)

    def test_export_plots_expected_files(self):
        files_to_data = {
            'a': radar.tree_data(make_tree()),
            'b': radar.tree_data(make_tree()),
        }
        radar.export_plots(self.tmpdir, files_to_data)
        self.assertItemsEqual(os.listdir(self.tmpdir),
                              ['radarChart.js', 'a.html', 'b.html'])

if __name__ == '__main__':
    unittest.main()
