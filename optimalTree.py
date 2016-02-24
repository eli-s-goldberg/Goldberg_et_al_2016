# To install requirements pip install -r requirements.txt
'''
This file does some interesting things. It imports the developed database, converts it to dimensionless parameters, and
then trains a decision tree based on those features. Here it is important to remember that we're not evaluating the
generic performance of the decision tree to predict the data outcome. We're using the decision tree as a method to
quantitatively investigate and breakdown the data... to see if we can disentangle the relationship between physicochem-
ical parameters and the retention behavior of nanomaterials.

Note that, becuase there are elements of stochasticity in the decision tree growing process, it can be difficult to
obtain the optimal decision tree (n.b. for most cases, there is never an optimal decision tree). Here, we cannot
guarentee optimality, but we can iteratively investigate the results of the decision tree and pick the best one.

Here we employ 5000 decision tree runs and report the index of the best one. This index value should be used to declare
the location of the output hierarchical JSON file (flareXX.json), which is output to the 
figures/decisionTreeVisualization/flare_reports folder. Once the index has been located, modify the appropriate variable
in the index.html file contained within teh decisionTreeVisualization folder and run it. This should present the tree
in all it's glory:)
'''
import multiprocessing

multiprocessing.cpu_count()
from helperFunctions import *
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import os
from sklearn import metrics
from sklearn import grid_search

from sklearn.externals.six import StringIO
from sklearn import tree  # old scikit learn 0.16.1
import pydot
import json
from sklearn.metrics import classification_report

# Set path
path = os.path.join('./')

# Set model iterations, cross folds, and randomness
iterations = 50  # Number of model iterations
kNum = 5  # Number of cross folds

# Set stack names for
importNames = []
for i in range(0, iterations):
    importNames.append('enmTransportData.xlsx')

# initialize containers for speed
rfecvGridScoresAll = pd.DataFrame()
bestParamsResults = pd.DataFrame()
dfGoodBadDataTrain = pd.DataFrame()
optimumLengthAll = pd.DataFrame()
outputScoresCV = pd.DataFrame()
testIndexTrack = pd.DataFrame()
trainIndexTrack = pd.DataFrame()
mseScoresTrack = pd.DataFrame()
featureImportanceStore = pd.DataFrame()
r2scoreHoldoutTrack = pd.DataFrame()
CVR2scoreTrack = pd.DataFrame()
rfecvGridScoresAll = pd.DataFrame()
optimumLengthAll = pd.DataFrame()
nameListAll = pd.DataFrame()
nameListAll = pd.DataFrame()
y_pred2Track = pd.DataFrame()
y_holdoutTrack = pd.DataFrame()
rfecvGridScoresWithRegNameTrack = pd.DataFrame()
contourTrack = pd.DataFrame()
xxTrack = pd.DataFrame()
yyTrack = pd.DataFrame()
y_randTest = pd.DataFrame()
f1_binary_average_score_track = []
f1_report = pd.DataFrame()

if __name__ == '__main__':  # wrap inside to prevent parallelize errors on windows.
    run = 0  # set initial run ID

    for name in importNames:  # Loop through all model iterations by looping through database names

        run = run + 1  # Allocate current run ID

        print run  # Print for convenience

        # import database as a function of path and iterated database name
        alphaDataset = pd.read_excel(os.path.join(path, 'transport_database', name))

        # Identify columns that are not needed for the assessment and drop
        dropColumnList = [
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

        alphaDataset = alphaDataset.drop(dropColumnList, 1)

        # remove any row without data (NA)
        alphaDataset = alphaDataset.dropna()

        # save the dataset for later inspection and use as refinedDataset
        alphaDataset.to_csv('refinedDataset.csv')

        # copy the refined dataset to a new variable.
        trainingData = alphaDataset.copy(deep=True)

        # drop the classification from the training data (that wouldn't be fun:)
        trainingData = trainingData.drop(['ObsRPShape'], 1)

        # Set the target data, copy into a new database and binarize inputs to exponential or nonexponential.
        targetData = pd.DataFrame(alphaDataset.ObsRPShape, columns=['ObsRPShape'])
        targetData = targetData.apply(binaryRPClassAssign, axis=1)

        # categorical feature evaluation: Step 1 create containers for feature names and dataframe for uniques.
        trainingCatFeatureNames = []
        trainingFeatureUniques = pd.DataFrame()

        # categorical feature evaluation: Step 2 copy features into a seperate database so you don't mess up the first
        trainDataFeatureInspect = trainingData.copy(deep=True)

        # set an if statment to check what the categorical features are.
        if name == 'enmTransportData.xlsx':
            trainingCatFeatureNames = [
                'NMId',
                'SaltType',
                'Coating',
                'TypeNOM'
            ]
        else:  # set your own if these are not they
            trainingCatFeatureNames = []

        # categorical feature evaluation: Step 3 loop through names, pull out and store uniques and factorize
        for features in trainingCatFeatureNames:
            tempdf = pd.DataFrame(trainDataFeatureInspect[features].factorize()[1], columns=[features])
            trainingFeatureUniques = pd.concat([trainingFeatureUniques, tempdf], axis=1)
            trainDataFeatureInspect[features] = trainDataFeatureInspect[features].factorize()[0]  # factorized data

        # create temporary dataframe and store names. Concat with the uniques in case the database name changes.
        tempdf2 = pd.DataFrame(columns=[name])
        trainingFeatureUniques = pd.concat([trainingFeatureUniques, tempdf2], axis=1)

        # print the unique features. This is to inspect the range of features, but also to get their assigned numbers.
        # print trainingFeatureUniques # Don't bother

        # Once again copy the factorized data from the above loop into the training data.
        trainingData = trainDataFeatureInspect.copy(deep=True)

        # Apply dimensionless number feature dimension reduction, start by assigning assumed constants (i.e., temp)
        # Note that this requires some data to be factorized. This will be changed in later versions, but now it
        # requires that we factorize, then apply and conform to dimensionless paramters, and then reencode to
        # recover categorical variables.
        trainingData['tempKelvin'] = 25 + 273.15
        trainingData['relPermValue'] = trainingData.apply(relPermittivity, axis=1)
        trainingData['N_r'] = trainingData.apply(dimAspectRatioAssign, axis=1)
        trainingData['N_a'] = trainingData.apply(attractionNumber, axis=1)
        trainingData['N_g'] = trainingData.apply(gravityNumber, axis=1)
        trainingData['N_Pe'] = trainingData.apply(dimPecletNumAssign, axis=1)
        trainingData['N_Lo'] = trainingData.apply(londonForce, axis=1)
        trainingData['D_l'] = trainingData.apply(debyeLength, axis=1)
        trainingData['N_Dl'] = trainingData.apply(debyeNumber, axis=1)
        trainingData['M_inj'] = trainingData.apply(massFlow, axis=1)
        trainingData['M_inj'] = 1e6 * trainingData['M_inj']  # in mg
        trainingData['N_as'] = trainingData.apply(porosityHappel, axis=1)
        trainingData['N_Z1'] = trainingData.apply(electrokinetic1, axis=1)
        trainingData['N_Z2'] = trainingData.apply(electrokinetic2, axis=1)
        trainingData['N_CA'] = trainingData['colLength'] / trainingData['colWidth']
        trainingData['ConcIn'] = 1e3 * trainingData['ConcIn']  # in mg/L
        trainingData['ConcHA'] = 1e3 * trainingData['ConcHA']  # in mg/L

        # put back categorical data encodes (see note above)
        trainingData['NMId'] = trainingData.apply(NMIDClassAssign, axis=1)
        trainingData['SaltType'] = trainingData.apply(SaltClassAssign, axis=1)
        trainingData['Coating'] = trainingData.apply(CoatingClassAssign, axis=1)
        trainingData['TypeNOM'] = trainingData.apply(TypeNOMClassAssign, axis=1)

        # Output a final copy of the training data for later use
        trainingData.to_csv('./trainingdataAll.csv', index=False, header=True)

        # Drop overlapping features - Combination assessment: temporary
        trainingData = trainingData.drop(['PublicationTitle', 'relPermValue', 'PartDensity', 'PvIn', 'Poros', 'D_l',
                                          'tempKelvin', 'colLength', 'colWidth',
                                          'PartDiam', 'CollecDiam',
                                          'Darcy', 'IonStr', 'SaltType', 'pH',
                                          'PartZeta', 'CollecZeta', 'PartIEP', 'Hamaker'
                                          ], 1)

        # More saving, post feature drop.
        targetData.to_csv('./targetdata.csv', index=False, header=True)
        trainingData.to_csv('./trainingdata.csv', index=False, header=True)

        # for use with the decision tree, get the remaining feature names
        feature_names = list(trainingData.columns.values)

        # encode the categorical variables using a one-hot scheme so they're correctly considered in decision tree
        ohe_enc = OneHotEncoder()
        trainingData, _, _ = one_hot_dataframe(trainingData, ['NMId', 'Coating', 'TypeNOM'], replace=True)

        # assign the target data as yAll and the training data as XAll. Notice that we train AND test on the same data.
        # This is not commmon, but we're employing the decision tree for a descriptive evaluation, not it's generic
        # prediction performance
        yAll = targetData.as_matrix()
        XAll = trainingData.as_matrix()

        # initialize the classifier
        clf = tree.DecisionTreeClassifier()

        # set a grid of parameters to investigate
        dpgrid = {'max_depth': [3, 4, 5],
                  'min_samples_leaf': [5, 6, 7, 8, 9, 10],
                  'max_features': [None, 'sqrt', 'log2'],  # , 'sqrt', 'log2'
                  'random_state': [None]
                  # 'class_weight': ['balanced'] # balance messes up the feature cound, use with caution
                  }

        # investigate the best possible set of parameters using a cross validation loop and the given grid
        gridSearch = grid_search.GridSearchCV(estimator=clf,
                                              cv=kNum,
                                              param_grid=dpgrid,
                                              n_jobs=-1)

        # call the grid search fit using the data
        gridSearch.fit(XAll, yAll)

        # store and print the best parameters
        bestParams = gridSearch.best_params_
        print bestParams

        # reinitialize and call the classifier with the best parameter
        clf = tree.DecisionTreeClassifier(**bestParams)
        clf.fit(XAll, yAll)

        # Evaluate the performance
        y_pred = clf.predict(XAll)  # predict the outcome by inputting all the data (this gives us the score)

        # calculate the score for the combined class (weighted), and then each class individually
        f1_binary_average_score = metrics.f1_score(yAll, y_pred, pos_label=None, average='weighted')
        f1_binary_average_score_exponential = metrics.f1_score(yAll, y_pred, pos_label=0)
        f1_binary_average_score_nonexponential = metrics.f1_score(yAll, y_pred, pos_label=1)

        # initialize scoring tracking dataframe to store the data
        f1_track = pd.DataFrame()
        f1_track['exponential'] = f1_binary_average_score_exponential,
        f1_track['nonexponential'] = f1_binary_average_score_nonexponential
        f1_track['average'] = f1_binary_average_score
        f1_report = f1_report.append(f1_track)
        f1_binary_average_score_track.append(f1_binary_average_score)

        # Compare the predictions to the truth directly and outut a file to inspect. 
        y_pred_frame = pd.DataFrame(y_pred, columns=['predicted'])
        y_truth_frame = pd.DataFrame(yAll, columns=['truth'])
        comparison = pd.concat([y_pred_frame, y_truth_frame], axis=1)
        comparison.to_csv('comparison.csv')

        ## The following section creates figures to visualize the decision tree as a PDF and to plot in D3 (java/html)
        # Feature elimination is not included here, but was included previously. This grabs only the names in the
        # remaining features.
        grabWorkingNames = [str(i) for i in list(trainingData)]

        # set the path to save the json representation.
        json_path = os.path.join(path, 'figures', 'decisionTreeVisualization', 'flare_reports', 'flare' + str(run) + '.json')

        dataTargetNames = ['exponential', 'nonexponential']
        r = rules(clf, grabWorkingNames, dataTargetNames)
        with open(json_path, 'w') as f:
            f.write(json.dumps(r))
        f.close()

        dot_data = StringIO()
        tree.export_graphviz(clf, out_file=dot_data,
                             feature_names=grabWorkingNames, impurity=True, rounded=True,
                             filled=True, label='all', leaves_parallel=True,
                             class_names=['exponential', 'nonexponential']
                             )

        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf(str(path) + '/output/trees/tree' + str(run) + '.pdf')
        class_reportPath = os.path.join(path, 'figures', 'decisionTreeVisualization', 'class_reports',
                                        'class_report' + str(run) + '.txt')
        # if there is an existing file with this name, clear it.
        file = open(class_reportPath, "w").close()

        # write code custom json because nothing else works.
        file = open(class_reportPath, "a")

        file.write(classification_report(yAll, y_pred, target_names=['exponential', 'nonexponential']))
        file.write('\n')
        file.close()

    ReportSavePath = os.path.join(path, 'figures', 'decisionTreeVisualization', 'DecisiontreeScores' + str(run) + '.csv')
    f1_report.to_csv(ReportSavePath)
    f1_report.reset_index(inplace=True)
    print f1_report.describe
    print "best performing decision tree index: ", f1_report['average'].argmax()
