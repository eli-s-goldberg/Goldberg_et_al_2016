# Installs the presubmit as a pre-commit hook.
cd "$(git rev-parse --show-toplevel)/.git/hooks"
ln -s ../../scripts/presubmit.sh pre-commit
