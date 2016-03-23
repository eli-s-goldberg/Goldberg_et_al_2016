#!/bin/bash
set -e
cd "$(git rev-parse --show-toplevel)"

if [[ $* == *--update_golden* ]]; then
    python pipeline_e2e_test.py --update_golden
fi
python -m unittest discover -p '*_test.py'
