# Installs the presubmit as a pre-commit hook.
cd "$(git rev-parse --show-toplevel)"
ln -s scripts/presubmit.sh .git/hooks/pre-commit
