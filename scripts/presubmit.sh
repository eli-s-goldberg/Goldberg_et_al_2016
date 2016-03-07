#!/bin/bash
# Run presubmit checks.
# Note: If this is enabled as a precommit hook, it can be by using --no-verify.
set -e
cd "$(git rev-parse --show-toplevel)"

echo Running lint...
# TODO(peterthenelson) Remove this blacklist once things are clean.
dirty_files=(
  ./class_test.py
  ./csv2flare2.py
  ./CV_calculate.py
  ./helper_functions.py
  ./histogramVisualization.py
  ./optimal_tree.py
)
pylint -r n $(comm -23 <(find . -iname '*.py'|sort) \
                       <(printf '%s\n' "${dirty_files[@]}"|sort))

echo Running tests...
python scripts/run_tests.py
