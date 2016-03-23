"""
Before running this file, run 'make_database.py' first. Here it is important to 
remember that we're not evaluating the generic performance of the decision tree 
to predict the data outcome. We're using the decision tree as a method to 
quantitatively investigate and breakdown the data... to see if we can disentangle 
the relationship between physicochemical parameters and the retention behavior of 
nanomaterials.

Note that, becuase there are elements of stochasticity in the decision tree
growing process, it can be difficult to obtain the optimal decision tree (n.b.
for most cases, there is never an optimal decision tree). Here, we cannot
guarentee optimality, but we can iteratively investigate the results of the
decision tree and pick the best one.

Here we employ many decision tree runs and report the index of the best one.
This index value should be used to declare the location of the output
hierarchical JSON file (flareXX.json), which is output to the
figures/decisionTreeVisualization/flare_reports folder. Once the index has been
located, modify the appropriate variable in the index.html file contained within
the decisionTreeVisualization folder and run it.
"""

import json
import os
import pandas as pd
import numpy as np
import pydot
from sklearn import metrics
from sklearn import grid_search
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import classification_report
from sklearn.cross_validation import StratifiedShuffleSplit

from helper_functions import (make_dirs, rules)

# Default database
TRAINING_PATH = os.path.join(
    os.path.dirname(__file__), 'transport_database', 'training_data.csv')
TARGET_PATH = os.path.join(
    os.path.dirname(__file__), 'transport_database', 'target_data.csv')

# Seed to use when running in deterministic mode.
_SEED = 666


# TODO(peterthenelson) Break up into functions
# TODO(peterthenelson) Use argparse module for flags
def main(path='.', training_path=TRAINING_PATH, target_path=TARGET_PATH,
         iterations=50, deterministic=False, stratified_holdout=True,
         holdout_size=0.15, crossfolds=5):
    """Find optimal decision tree, write output files.

    Parameters
    ----------
    path : str
        Path to output directory.
    training_path : str
        Path to training data csv.
    target_path : str
        Path to target data csv.
    iterations : int
        Number of runs of fitting the model.
    deterministic : bool
        Turn off randomness (for testing).
    stratified_holdout : bool
        Turn off the use of a stratified holdout set. Use with caution: 
        False = train and test on the same data (ok for description, but 
        not prediction).
    holdout_size : float
        The percentage of the database not employed for training. 
    crossfolds : int
        Number of folds for crossvalidation.
    """
    # TODO(peterthenelson) This is a dumb thing to do. Name and location should
    # be bundled into a config object, rather than trying to derive it from the
    # path (or one of them).
    database_basename = os.path.basename(training_path)
    # Everything goes under this subdirectory.
    path = os.path.join(path, 'classifier')

    # Loop through all model interactions by looping through database names
    run = 0
    f1_binary_average_score_track = []
    f1_report = pd.DataFrame()

    target_data = np.squeeze(pd.read_csv(target_path))
    training_data = pd.read_csv(training_path)

    for run in xrange(iterations):
        print run  # Print for convenience  
        y_train = np.array(target_data)
        x_train = training_data.as_matrix()

        # assign the target data as y_all and the training data as x_all. Notice
        # that we train AND test on the same data. This is not commmon, but
        # we're employing the decision tree for a descriptive evaluation, not
        # its generic prediction performance

        if stratified_holdout:
            random_state = _SEED if deterministic else None
            sss = StratifiedShuffleSplit(
                y_train, n_iter=1, test_size=holdout_size,
                random_state=random_state)

            for train_index, test_index in sss:
                x_train, x_holdout = x_train[train_index], x_train[test_index]
                y_train, y_holdout = y_train[train_index], y_train[test_index]

            x_train_or_holdout = x_holdout
            y_train_or_holdout = y_holdout

            # if you want to seperate training data into holdout set to examine performance.
            x_train_or_holdout = x_train
            y_train_or_holdout = y_train

        # initialize the classifier
        clf = tree.DecisionTreeClassifier()

        # optimize classifier by brute-force parameter investigation
        dpgrid = {'max_depth': [3,4,5],
                  'min_samples_leaf': [11,12,13],
                  'max_features': [None, 'sqrt', 'log2'],
                  'random_state': [_SEED] if deterministic else [None]
                  }

        # investigate the best possible set of parameters using a cross
        # validation loop and the given grid. The cross-validation does not do
        # random shuffles, but the estimator does use randomness (and
        # takes random_state via dpgrid).
        grid_searcher = grid_search.GridSearchCV(estimator=clf, cv=crossfolds,
                                                 param_grid=dpgrid, n_jobs=-1)

        # call the grid search fit using the data
        grid_searcher.fit(x_train, y_train)

        # store and print the best parameters
        best_params = grid_searcher.best_params_

        # reinitialize and call the classifier with the best parameter
        clf = tree.DecisionTreeClassifier(**best_params)
        clf.fit(x_train, y_train)

        # Evaluate external performance (how well does
        # the trained model classify the holdout?)
        y_pred = clf.predict(x_train_or_holdout)

        # calculate the score for the combined class (weighted), and then
        # each class individually
        f1_binary_average_score = metrics.f1_score(
            y_train_or_holdout, y_pred, pos_label=None, average='weighted')
        f1_binary_average_score_exp = metrics.f1_score(
            y_train_or_holdout, y_pred, pos_label=0)
        f1_binary_average_score_nonexp = metrics.f1_score(
            y_train_or_holdout, y_pred, pos_label=1)

        # Compare the predictions to the truth directly and outut a file
        # to inspect.
        y_pred_frame = pd.DataFrame(y_pred, columns=['predicted'])
        y_truth_frame = pd.DataFrame(y_train_or_holdout, columns=['truth'])
        comparison = pd.concat([y_pred_frame, y_truth_frame], axis=1)
        comparison.to_csv(os.path.join(path, 'comparison.csv'))

        # initialize scoring tracking dataframe to store the data
        f1_track = pd.DataFrame()
        f1_track['exponential'] = f1_binary_average_score_exp,
        f1_track['nonexponential'] = f1_binary_average_score_nonexp
        f1_track['average'] = f1_binary_average_score
        f1_report = f1_report.append(f1_track)  # pylint:disable=redefined-variable-type
        f1_binary_average_score_track.append(f1_binary_average_score)

        # The following section creates figures to visualize the decision tree
        # as a PDF and to plot in D3 (java/html). Feature elimination is not
        # included here, but was included previously. This grabs only the names
        # in the remaining features.
        grab_working_names = [str(i) for i in list(training_data)]

        # set the path to save the json representation.
        json_dir = os.path.join(path, 'flare')
        make_dirs(json_dir)
        json_path = os.path.join(json_dir, 'flare%d.json' % (run + 1))

        data_target_names = ['exponential', 'nonexponential']

        tree_rules = rules(clf, grab_working_names, data_target_names)
        with open(json_path, 'w') as outf:
            outf.write(json.dumps(tree_rules))

        dot_data = StringIO()
        tree.export_graphviz(clf, out_file=dot_data,
                             feature_names=grab_working_names, impurity=True,
                             rounded=True, filled=True, label='all',
                             leaves_parallel=True,
                             class_names=['exponential', 'nonexponential'])

        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        make_dirs(os.path.join(path, 'models'))
        graph.write_pdf(os.path.join(path, 'models/%d.pdf' % (run + 1)))
        class_report_dir = os.path.join(path, 'class_reports')
        make_dirs(class_report_dir)
        class_report_path = os.path.join(class_report_dir,
                                         'class_report%d.txt' % (run + 1))
        with open(class_report_path, "w") as outf:
            outf.write(classification_report(
                y_train_or_holdout, y_pred, target_names=['exponential', 'nonexponential']))
            outf.write('\n')

    report_save_path = os.path.join(path, 'classifier', 'scores%d.csv' % (run + 1))
    f1_report.to_csv(report_save_path)
    f1_report.reset_index(inplace=True)
    print f1_report.describe()
    print "best performing decision tree index: ", f1_report['average'].argmax()


if __name__ == '__main__':  # wrap inside to prevent parallelize errors on windows.
    main()
