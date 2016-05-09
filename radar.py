"""Creates radar plots for decision trees."""

import json
import os.path
import random
import shutil
import string

import helper_functions

_JS_FILE = os.path.join(
    os.path.dirname(__file__), 'templates', 'radarChart.js')
_HTML_TEMPLATE = os.path.join(
    os.path.dirname(__file__), 'templates', 'radarChart.html')
_CHECKBOX_TEMPLATE = os.path.join(
    os.path.dirname(__file__), 'templates', 'checkboxes.html')

def tree_data(tree_classifier):
    """Extract data from a decision tree for use in a radar plot.

    Parameters
    ----------
    tree_classifier : DecisionTreeClassifier
        A tree that has already been fit.

    Returns
    -------
    list of lists of dicts
        Data appropriate to jsonify and use with radar plot.
    """
    # TODO(peterthenelson) Don't actually know what to do. Here's some data
    # that's the right format.
    _ = tree_classifier
    axes = ["ConcHA", "ConcIn", "M_inj", "N_CA", "N_Dl", "N_Lo", "N_Pe", "N_Z1",
            "N_Z2", "N_a", "N_as", "N_g", "N_r"]
    data = []
    for _ in xrange(2):
        data.append([
            {"axis": "CV for " + axis, "value": random.uniform(0.0, 5.0)}
            for axis in axes])
    return data


def export_plots(dir_path, files_to_data):
    """Export radar data as html and javascript files.

    Parameters
    ----------
    dir_path : str
        Path to a directory where output files will be written to.
    files_to_data : dict
        Keys are the plot names (which will be used in file names), and the
        values are the corresponding radar plot data values (see tree_data).
    """
    helper_functions.make_dirs(dir_path)
    shutil.copy(_JS_FILE, dir_path)
    with open(_HTML_TEMPLATE) as f:
        html_template = string.Template(f.read())
    with open(_CHECKBOX_TEMPLATE) as f:
        checkbox_template = string.Template(f.read())
    for f, d in files_to_data.iteritems():
        axes = {item['axis'] for ring in d for item in ring}
        checkboxes = ''
        for i, axis in enumerate(sorted(axes)):
            checkboxes += checkbox_template.substitute(index=i, axis=axis)
        with open(os.path.join(dir_path, f + '.html'), 'w') as html:
            html.write(html_template.substitute(
                checkboxes=checkboxes, data=json.dumps(d, indent=2)))
