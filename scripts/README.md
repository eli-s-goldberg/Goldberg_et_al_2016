Scripts for use by developers. They must be run from somewhere in the repo.

- `install_presubmit.sh`: Installs the pre-commit hook. Afterwards, every time
you `git commit` in the repo, the `presubmit.sh` script will run, and the commit
will only succeed if the presubmit passes. You can override this check with
`--no-verify`.
- `lint.sh`: Runs pylint on all of the python files. You need pylint installed
in order for this to work. See `.pylintrc` for special repo-specific settings.
- `presubmit.sh`: Runs the presubmit. This includes a check that there are no
unstaged changes or untracked (and non-ignored) files, a check that lint passes,
and a check that tests pass.
- `run_tests.sh`: Runs the tests.
