#!/bin/bash
# Run presubmit checks.
# Note: If this is enabled as a precommit hook, it can be by using --no-verify.
set -e
cd "$(git rev-parse --show-toplevel)"

echo Running lint...
scripts/lint.sh

echo Running tests...
scripts/run_tests.sh
