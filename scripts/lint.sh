# TODO(peterthenelson) Remove this blacklist once things are clean.
dirty_files=(
  ./class_test.py
  ./csv2flare2.py
  ./CV_calculate.py
  ./histogramVisualization.py
  ./optimal_tree.py
)
pylint --rcfile=.pylintrc $(comm -23 <(find . -iname '*.py'|sort) \
                          <(printf '%s\n' "${dirty_files[@]}"|sort))
