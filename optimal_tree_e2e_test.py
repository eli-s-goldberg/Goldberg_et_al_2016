"""End-to-end test for the optimalTree pipeline.

This is a characterization test in Michael Feathers' sense of the term. It's not
something I plan to keep around; it just provides some assurance that
refactoring is not actually resulting in functional changes.
"""

import difflib
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

    def assert_dirs_equal(self, dir1, dir2):
        """Assert that directories are equal, printing out diff.

        Helper functions adapted from filecmp.dircmp source.
        """
        dc = filecmp.dircmp(dir1, dir2)
        self.assertTrue(
            self._dir_eq(dc),
            msg=('Directories %s and %s differ!' % (dir1, dir2)))

    def _dir_eq(self, dc):
        eq = True
        if dc.left_only:
            eq = False
            print 'Only in', dc.left, ':', sorted(dc.left_only)
        if dc.right_only:
            eq = False
            print 'Only in', dc.right, ':', sorted(dc.right_only)
        if dc.diff_files:
            eq = False
            dc.diff_files.sort()
            print 'Differing files :', dc.diff_files
            for f in dc.diff_files:
                fname1 = os.path.join(dc.left, f)
                with open(fname1) as f1:
                    lines1 = f1.readlines()
                fname2 = os.path.join(dc.right, f)
                with open(fname2) as f2:
                    lines2 = f2.readlines()
                for line in difflib.unified_diff(
                    lines1, lines2, fromfile=fname1, tofile=fname2):
                    print line,
        if dc.funny_files:
            eq = False
            print 'Trouble with common files :', sorted(dc.funny_files)
        if dc.common_funny:
            eq = False
            print 'Common funny cases :', sorted(dc.common_funny)
        return eq

    def _full_closure_eq(dc):
        eq = self._dir_eq(dc)
        for sd in dc.subdirs.itervalues():
            eq = eq and self._full_closure_eq(sd)
        return eq

    def test_optimal_tree(self):
        optimalTree.main(path=self.tmpdir, iterations=2, deterministic=True)
        self.assert_dirs_equal(TESTDATA_PATH, self.tmpdir)

if __name__ == '__main__':
    unittest.main()
