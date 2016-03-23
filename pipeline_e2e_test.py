"""End-to-end test for the full pipeline.

This is a characterization test in Michael Feathers' sense of the term. It's not
something I plan to keep around; it just provides some assurance that
refactoring is not actually resulting in functional changes.
"""

import difflib
import filecmp
import os
import os.path
import shutil
import sys
import tempfile
import unittest

import histogram_visualization
import make_database
import optimal_tree

TESTDATA_PATH = os.path.join(os.path.dirname(__file__), 'testdata')
FULL_DIFF_FORMATS = ['.csv', '.txt']

def dir_eq(dircmp, recursive=True):
    """Compare two directories for equality and print out diffs.

    Adapted from filecmp.dircmp source.

    Parameters
    ----------
    dircmp : dircmp
    recursive : bool, optional

    Returns
    -------
    bool
        Are the directories (and optionally subdirectories) equal, including
        the contents of their files?

    """
    equal = True
    if dircmp.left_only:
        equal = False
        print 'Only in', dircmp.left, ':', sorted(dircmp.left_only)
    if dircmp.right_only:
        equal = False
        print 'Only in', dircmp.right, ':', sorted(dircmp.right_only)
    if dircmp.diff_files:
        equal = False
        dircmp.diff_files.sort()
        print 'Differing files :', dircmp.diff_files
        for fname in dircmp.diff_files:
            _, ext = os.path.splitext(fname)
            if ext.lower() not in FULL_DIFF_FORMATS:
                print '%s: full diffs not displayed for file type %s' % (
                    fname, ext)
                continue
            # TODO(#31) Don't do diffs w/a nonexistent file!
            fname1 = os.path.join(dircmp.left, fname)
            with open(fname1) as file1:
                lines1 = file1.readlines()
            fname2 = os.path.join(dircmp.right, fname)
            with open(fname2) as file2:
                lines2 = file2.readlines()
            for line in difflib.unified_diff(
                    lines1, lines2, fromfile=fname1, tofile=fname2):
                print line,
    if dircmp.funny_files:
        equal = False
        print 'Trouble with common files :', sorted(dircmp.funny_files)
    if dircmp.common_funny:
        equal = False
        print 'Common funny cases :', sorted(dircmp.common_funny)
    if recursive:
        for subdir in dircmp.subdirs.itervalues():
            # NOTE: ordering important
            equal = dir_eq(subdir, recursive=True) and equal
    return equal

def update_golden():
    """Update the golden files in the testdata directory."""
    confirm = raw_input('Do you really want to rebuild the golden files?\n'
                        'Did you actually look at the diffs? (y/n): ')
    if confirm.strip().lower() not in ['y', 'yes']:
        return
    print 'Deleting testdata'
    shutil.rmtree(TESTDATA_PATH)
    os.makedirs(TESTDATA_PATH)
    print 'Rebuilding testdata'
    run_pipeline(TESTDATA_PATH)

def run_pipeline(path):
    """Run the pipeline, putting output into specified directory."""
    make_database.main(path=path)
    training = os.path.join(path, 'data/training_data.csv')
    target = os.path.join(path, 'data/target_data.csv')
    optimal_tree.main(training_path=training, target_path=target, path=path,
                      iterations=2, deterministic=True)
    histogram_visualization.main(training_path=training, target_path=target, path=path)

class PipelineE2ETest(unittest.TestCase):
    """End-to-end test for whole pipeline."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def assert_dirs_equal(self, dir1, dir2):
        """Assert that directories are equal, printing out diff."""
        self.assertTrue(
            dir_eq(filecmp.dircmp(dir1, dir2)),
            msg=('Directories %s and %s differ!' % (dir1, dir2)))

    def test_pipeline(self):
        """Compare full run of pipeline against a golden output dir."""
        run_pipeline(self.tmpdir)
        self.assert_dirs_equal(TESTDATA_PATH, self.tmpdir)

if __name__ == '__main__':
    # Poor man's argparse
    if '--update_golden' in sys.argv:
        update_golden()
    else:
        unittest.main()
