"""End-to-end test for the optimalTree pipeline.

This is a characterization test in Michael Feathers' sense of the term. It's not
something I plan to keep around; it just provides some assurance that
refactoring is not actually resulting in functional changes.
"""

import filecmp
import os
import os.path
import shutil
import tempfile
import unittest

import optimalTree

TESTDATA_PATH = os.path.join(os.path.dirname(__file__), 'testdata')

class OptimalTreeE2ETest(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_optimal_tree(self):
        optimalTree.main(path=self.tmpdir, iterations=2, deterministic=True)
        dc = filecmp.dircmp(TESTDATA_PATH, self.tmpdir)
        # TODO(peterthenelson) This needs to be an assertion instead of a
        # printed summary, but this means I neeed to reimplement the recursive
        # checking logic.
        dc.report_full_closure()

if __name__ == '__main__':
    unittest.main()
