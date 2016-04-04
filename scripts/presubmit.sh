#!/bin/bash
# Run presubmit checks.
# Note: If this is enabled as a precommit hook, it can be by using --no-verify.
set -e
cd "$(git rev-parse --show-toplevel)"

echo Checking for unstaged changes...
if ! UNSTAGED=$(git diff --name-only --exit-code)
then
  printf "Unstaged changes!:\n%s\n" "$UNSTAGED"
  exit 1
fi

echo Checking for untracked files...
UNTRACKED=$(git ls-files --other --exclude-standard)
if [[ "$UNTRACKED" ]]
then
  printf "Untracked files!:\n%s\n" "$UNTRACKED"
  exit 1
fi

echo Running lint...
scripts/lint.sh

echo Running tests...
scripts/run_tests.sh
