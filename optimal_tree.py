"""
This file does some interesting things. It imports the developed database,
converts it to dimensionless parameters, and then trains a decision tree based
on those features. Here it is important to remember that we're not evaluating
the generic performance of the decision tree to predict the data outcome. We're
using the decision tree as a method to quantitatively investigate and breakdown
the data... to see if we can disentangle the relationship between physicochem-
ical parameters and the retention behavior of nanomaterials.

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
teh decisionTreeVisualization folder and run it. This should present the tree in
all it's glory :)
"""

import json
import os
import pandas as pd
import pydot
from sklearn import metrics
from sklearn import grid_search
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import classification_report

from helper_functions import (
    make_dirs, binaryRPClassAssign, dimAspectRatioAssign, dimPecletNumAssign,
    attractionNumber, gravityNumber, debyeLength, massFlow, electrokinetic1,
    electrokinetic2, relPermittivity, londonForce, porosityHappel, debyeNumber,
    rules, NMIDClassAssign, SaltClassAssign, CoatingClassAssign,
    TypeNOMClassAssign, one_hot_dataframe)

# TODO(peterthenelson) Break up into functions
# TODO(peterthenelson) Use argparse module for flags
def main(path='.', iterations=50, deterministic=False, crossfolds=5):
    """Find optimal decision tree, write output files.

    Parameters
    ----------
    path : str
        Path to output directory.
    iterations : int
        Number of runs of fitting the model.
    deterministic : bool
        Turn off randomness (for testing).
    crossfolds : int
        Number of folds for crossvalidation.

    """
    # TODO(peternelson) Is this supposed to be configurable?
    import_names = ['enmTransportData.xlsx'] * iterations
    # Loop through all model interactions by looping through database names
    run = 0
    f1_binary_average_score_track = []
    f1_report = pd.DataFrame()
    for run, name in enumerate(import_names):
        print run  # Print for convenience

        # import database as a function of path and iterated database name
        alpha_dataset = pd.read_excel(os.path.join(
            os.path.dirname(__file__), 'transport_database', name))

        # Identify columns that are not needed for the assessment and drop
        drop_column_list = [
            'ProfileID',
            'ColumnLWRatio',
            'mbEffluent',
            'mbRetained',
            'mbEffluent_norm',
            'mbRetained_norm',
            'Dispersivity',
            'Notes1',
            'Notes2',
            'Notes3',
            'Notes4',
            'Notes5']

        alpha_dataset = alpha_dataset.drop(drop_column_list, 1)

        # remove any row without data (NA)
        alpha_dataset = alpha_dataset.dropna()

        # save the dataset for later inspection and use as refinedDataset
        alpha_dataset.to_csv(os.path.join(path, 'refinedDataset.csv'))

        # copy the refined dataset to a new variable.
        training_data = alpha_dataset.copy(deep=True)

        # drop the classification from the training data (that wouldn't be fun:)
        training_data = training_data.drop(['ObsRPShape'], 1)

        # Set the target data, copy into a new database and binarize
        # inputs to exponential or nonexponential.
        target_data = pd.DataFrame(
            alpha_dataset.ObsRPShape, columns=['ObsRPShape']).apply(
                binaryRPClassAssign, axis=1)

        # categorical feature evaluation: Step 1 create containers for
        # feature names and dataframe for uniques.
        training_cat_feature_names = []
        training_feature_uniques = pd.DataFrame()

        # categorical feature evaluation: Step 2 copy features into a seperate
        # database so you don't mess up the first
        train_data_feature_inspect = training_data.copy(deep=True)

        # set an if statment to check what the categorical features are.
        if name == 'enmTransportData.xlsx':
            training_cat_feature_names = [
                'NMId',
                'SaltType',
                'Coating',
                'TypeNOM'
            ]
        else:  # set your own if these are not they
            training_cat_feature_names = []

        # categorical feature evaluation: Step 3 loop through names, pull out
        # and store uniques and factorize
        for features in training_cat_feature_names:
            tempdf = pd.DataFrame(
                train_data_feature_inspect[features].factorize()[1],
                columns=[features])
            training_feature_uniques = pd.concat( # pylint:disable=redefined-variable-type
                [training_feature_uniques, tempdf], axis=1)
            train_data_feature_inspect[features] = (
                train_data_feature_inspect[features].factorize()[0])

        # create temporary dataframe and store names. Concat with the uniques
        # in case the database name changes.
        tempdf2 = pd.DataFrame(columns=[name])
        training_feature_uniques = pd.concat(
            [training_feature_uniques, tempdf2], axis=1)

        # Once again copy the factorized data from the above loop into the
        # training data.
        training_data = train_data_feature_inspect.copy(deep=True)

        # Apply dimensionless number feature dimension reduction, start by
        # assigning assumed constants (i.e., temp). Note that this requires some
        # data to be factorized. This will be changed in later versions, but now
        # it requires that we factorize, then apply and conform to dimensionless
        # parameters, and then reencode to recover categorical variables.
        training_data['tempKelvin'] = 25 + 273.15
        training_data['relPermValue'] = training_data.apply(relPermittivity,
                                                            axis=1)
        training_data['N_r'] = training_data.apply(dimAspectRatioAssign, axis=1)
        training_data['N_a'] = training_data.apply(attractionNumber, axis=1)
        training_data['N_g'] = training_data.apply(gravityNumber, axis=1)
        training_data['N_Pe'] = training_data.apply(dimPecletNumAssign, axis=1)
        training_data['N_Lo'] = training_data.apply(londonForce, axis=1)
        training_data['D_l'] = training_data.apply(debyeLength, axis=1)
        training_data['N_Dl'] = training_data.apply(debyeNumber, axis=1)
        training_data['M_inj'] = training_data.apply(massFlow, axis=1)
        training_data['M_inj'] = 1e6 * training_data['M_inj']  # in mg
        training_data['N_as'] = training_data.apply(porosityHappel, axis=1)
        training_data['N_Z1'] = training_data.apply(electrokinetic1, axis=1)
        training_data['N_Z2'] = training_data.apply(electrokinetic2, axis=1)
        training_data['N_CA'] = (
            training_data['colLength'] / training_data['colWidth'])
        training_data['ConcIn'] = 1e3 * training_data['ConcIn']  # in mg/L
        training_data['ConcHA'] = 1e3 * training_data['ConcHA']  # in mg/L

        # put back categorical data encodes (see note above)
        training_data['NMId'] = training_data.apply(NMIDClassAssign, axis=1)
        training_data['SaltType'] = training_data.apply(SaltClassAssign, axis=1)
        training_data['Coating'] = training_data.apply(CoatingClassAssign, axis=1)
        training_data['TypeNOM'] = training_data.apply(TypeNOMClassAssign, axis=1)

        # Output a final copy of the training data for later use
        training_data.to_csv(
            os.path.join(path, 'trainingdataAll.csv'), index=False, header=True)

        # Drop overlapping features - Combination assessment: temporary
        training_data = training_data.drop(
            ['PublicationTitle', 'relPermValue', 'PartDensity', 'PvIn', 'Poros',
             'D_l', 'tempKelvin', 'colLength', 'colWidth', 'PartDiam',
             'CollecDiam', 'Darcy', 'IonStr', 'SaltType', 'pH', 'PartZeta',
             'CollecZeta', 'PartIEP', 'Hamaker'], 1)

        # More saving, post feature drop.
        target_data.to_csv(os.path.join(path, 'targetdata.csv'),
                           index=False, header=True)
        training_data.to_csv(os.path.join(path, 'trainingdata.csv'),
                             index=False, header=True)

        # encode the categorical variables using a one-hot scheme so they're
        # correctly considered in decision tree
        training_data, _, _ = one_hot_dataframe(
            training_data, ['NMId', 'Coating', 'TypeNOM'], replace=True)

        # assign the target data as y_all and the training data as x_all. Notice
        # that we train AND test on the same data. This is not commmon, but
        # we're employing the decision tree for a descriptive evaluation, not
        # its generic prediction performance
        y_all = target_data.as_matrix()
        x_all = training_data.as_matrix()

        # initialize the classifier
        clf = tree.DecisionTreeClassifier()

        # set a grid of parameters to investigate
        random_state = [None]
        if deterministic:
            random_state = [666]
        dpgrid = {'max_depth': [3, 4, 5],
                  'min_samples_leaf': [5, 6, 7, 8, 9, 10],
                  'max_features': [None, 'sqrt', 'log2'],
                  'random_state': random_state,
                  # balance messes up the feature cound, use with caution
                  # 'class_weight': ['balanced']
                 }

        # investigate the best possible set of parameters using a cross
        # validation loop and the given grid. The cross-validation does not do
        # random shuffles, but the estimator does use randomness (and
        # takes random_state via dpgrid).
        grid_searcher = grid_search.GridSearchCV(estimator=clf, cv=crossfolds,
                                                 param_grid=dpgrid, n_jobs=-1)

        # call the grid search fit using the data
        grid_searcher.fit(x_all, y_all)

        # store and print the best parameters
        best_params = grid_searcher.best_params_
        print best_params

        # reinitialize and call the classifier with the best parameter
        clf = tree.DecisionTreeClassifier(**best_params)
        clf.fit(x_all, y_all)

        # Evaluate the performance
        y_pred = clf.predict(x_all)

        # calculate the score for the combined class (weighted), and then
        # each class individually
        f1_binary_average_score = metrics.f1_score(
            y_all, y_pred, pos_label=None, average='weighted')
        f1_binary_average_score_exp = metrics.f1_score(
            y_all, y_pred, pos_label=0)
        f1_binary_average_score_nonexp = metrics.f1_score(
            y_all, y_pred, pos_label=1)

        # initialize scoring tracking dataframe to store the data
        f1_track = pd.DataFrame()
        f1_track['exponential'] = f1_binary_average_score_exp,
        f1_track['nonexponential'] = f1_binary_average_score_nonexp
        f1_track['average'] = f1_binary_average_score
        f1_report = f1_report.append(f1_track) # pylint:disable=redefined-variable-type
        f1_binary_average_score_track.append(f1_binary_average_score)

        # Compare the predictions to the truth directly and outut a file
        # to inspect.
        y_pred_frame = pd.DataFrame(y_pred, columns=['predicted'])
        y_truth_frame = pd.DataFrame(y_all, columns=['truth'])
        comparison = pd.concat([y_pred_frame, y_truth_frame], axis=1)
        comparison.to_csv(os.path.join(path, 'comparison.csv'))

        # The following section creates figures to visualize the decision tree
        # as a PDF and to plot in D3 (java/html). Feature elimination is not
        # included here, but was included previously. This grabs only the names
        # in the remaining features.
        grab_working_names = [str(i) for i in list(training_data)]

        # set the path to save the json representation.
        json_dir = os.path.join(
            path, 'figures', 'decisionTreeVisualization', 'flare_reports')
        make_dirs(json_dir)
        json_path = os.path.join(json_dir, 'flare%d.json' % (run+1))

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
        make_dirs(os.path.join(path, 'output/trees/tree'))
        graph.write_pdf(
            os.path.join(path, 'output/trees/tree/%d.pdf' % (run+1)))
        class_report_dir = os.path.join(
            path, 'figures', 'decisionTreeVisualization', 'class_reports')
        make_dirs(class_report_dir)
        class_report_path = os.path.join(class_report_dir,
                                         'class_report%d.txt' % (run+1))
        with open(class_report_path, "w") as outf:
            outf.write(classification_report(
                y_all, y_pred, target_names=['exponential', 'nonexponential']))
            outf.write('\n')

    report_save_path = os.path.join(
        path, 'figures', 'decisionTreeVisualization',
        'DecisiontreeScores%d.csv' % (run+1))
    f1_report.to_csv(report_save_path)
    f1_report.reset_index(inplace=True)
    print f1_report.describe()
    print "best performing decision tree index: ", f1_report['average'].argmax()

if __name__ == '__main__':  # wrap inside to prevent parallelize errors on windows.
    main()
