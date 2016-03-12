import multiprocessing
import numpy as np

from pandas import *
import csv, itertools, json
import math

multiprocessing.cpu_count()
from helperFunctions import *
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import metrics
from sklearn import grid_search
from sklearn.externals.six import StringIO
from sklearn import tree  # old scikit learn 0.16.1
import pydot
import json
import re
from sklearn.metrics import classification_report

# Set path
# path = '/Users/future/Google Drive/phd_root/programming_root/enmTransportPrediction/'
path = os.path.join('./')

# Set model iterations, cross folds, and randomness
iterations = 1  # Number of model iterations
kNum = 5  # Number of cross folds

# Set stack names for
importNames = []
for i in range(0, iterations):
    importNames.append('enmTransportData_edit.xlsx')

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
        targetData = targetData.apply(binary_rp_class_assignPlotClass, axis=1)

        # categorical feature evaluation: Step 1 create containers for feature names and dataframe for uniques.
        trainingCatFeatureNames = []
        trainingFeatureUniques = pd.DataFrame()

        # categorical feature evaluation: Step 2 copy features into a seperate database so you don't mess up the first
        trainDataFeatureInspect = trainingData.copy(deep=True)

        # set an if statment to check what the categorical features are.
        if name == 'enmTransportData_edit.xlsx':
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
        trainingData['relPermValue'] = trainingData.apply(rel_permittivity, axis=1)
        trainingData['N_r'] = trainingData.apply(dim_aspect_ratio_assign, axis=1)
        trainingData['N_a'] = trainingData.apply(attraction_number, axis=1)
        trainingData['N_g'] = trainingData.apply(gravity_number, axis=1)
        trainingData['N_Pe'] = trainingData.apply(dim_peclet_num_assign, axis=1)
        trainingData['N_Lo'] = trainingData.apply(london_force, axis=1)
        trainingData['D_l'] = trainingData.apply(debye_length, axis=1)
        trainingData['N_Dl'] = trainingData.apply(debye_number, axis=1)
        trainingData['M_inj'] = trainingData.apply(mass_flow, axis=1)
        trainingData['M_inj'] = 1e6 * trainingData['M_inj']  # in mg
        trainingData['N_as'] = trainingData.apply(porosity_happel, axis=1)
        trainingData['N_Z1'] = trainingData.apply(electrokinetic1, axis=1)
        trainingData['N_Z2'] = trainingData.apply(electrokinetic2, axis=1)
        trainingData['N_CA'] = trainingData['colLength'] / trainingData['colWidth']
        trainingData['ConcIn'] = 1e3 * trainingData['ConcIn']  # in mg/L
        trainingData['ConcHA'] = 1e3 * trainingData['ConcHA']  # in mg/L

        # put back categorical data encodes (see note above)
        trainingData['NMId'] = trainingData.apply(nmid_class_assign, axis=1)
        trainingData['SaltType'] = trainingData.apply(salt_class_assign, axis=1)
        trainingData['Coating'] = trainingData.apply(coating_class_assign, axis=1)
        trainingData['TypeNOM'] = trainingData.apply(type_nom_class_assign, axis=1)

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
        trainingData['classification'] = targetData
        # print trainingData.head()
        trainingData['ConcHA'].hist(by=trainingData['classification'], bins=10)
        # bigData = pd.groupby(trainingData,['N_Dl','classification']).size()


        binNumber = 10
        ParameterList = [
            'N_Lo',  # 13 bins
            'M_inj',  # 26 bins
            'N_as',  # 28 bins
            'N_Z2',  # 21 bins
            'N_CA',  # 9 bins
            'ConcIn',
            'ConcHA',
                      'N_Z1'
        ]
        binNumberDef = [13, 26, 28, 21, 9, 10, 10,8]

        for parameter in zip(ParameterList, binNumberDef):

            bigData = pd.DataFrame()
            bigData[parameter[0]] = trainingData[parameter[0]]
            bigData['classification'] = trainingData['classification']
            low = math.floor(bigData[parameter[0]].min())
            high = math.ceil(bigData[parameter[0]].max())
            # print high, parameter[0]
            bins = np.linspace(low, high, parameter[1], endpoint=True)
            expon = bigData[bigData['classification'].isin(['exponential'])]
            expon.columns = [str(parameter[0]), 'exponential']

            exponGrouped = expon.groupby(
            pd.cut(expon[str(parameter[0])], bins, precision=0)).size().reset_index() # bins? with parameter[1]
            exponGrouped.columns = [str(parameter[0]), 'exponential']

            nonexpon = bigData[bigData['classification'].isin(['nonexponential'])]
            nonexpon.columns = [str(parameter[0]), 'nonexponential']

            nonexponGrouped = nonexpon.groupby(pd.cut(nonexpon[str(parameter[0])], bins, precision=0)).size().reset_index()
            nonexponGrouped.columns = [str(parameter[0]), 'nonexponential']
            CombinedGrouped = pd.concat([exponGrouped, nonexponGrouped['nonexponential']], axis=1)

            pathIter = os.path.join('figures', 'histograms', 'tmps', str('tmp' + parameter[0] + 'tmp.csv'))

            file = open(pathIter, "w").close()  # if there is an existing file with this name, clear it.
            file = open(pathIter, "a")
            file.write(str(parameter[0] + ',' + 'exponential,nonexponential\n'))
            import fileinput

            for j in range(0, len(bins) - 1):  # when we get a range, there's always one left so -1
                if j < len(bins) - 1:
                    for i in range(0, 3):
                        if i == 0:
                            a = 0
                            a = CombinedGrouped.ix[j][i][1:-1]
                            file.write(str(a + ','))
                        elif i < 2:
                            b = 0
                            b = CombinedGrouped.ix[j][i]
                            file.write( str(b) + ',')
                        else:
                            b = 0
                            b = CombinedGrouped.ix[j][i]
                            file.write(str(b))
                    file.write('\n')
                else:
                    for i in range(0, 3):
                        if i == 0:
                            a = 0
                            a = CombinedGrouped.ix[j][i][1:-1]
                            file.write(str('' + a))
                        elif i < 2:
                            b = 0
                            b = CombinedGrouped.ix[j][i]
                            file.write(str(b) + ',')
                        else:
                            b = 0
                            b = CombinedGrouped.ix[j][i]
                            file.write(str(b))
                    file.write('\n')
            file.close()

            file = open(pathIter, "r")
            pathIterTMP = os.path.join('figures', 'histograms', str(parameter[0] + '.csv'))
            filewrite = open(pathIterTMP, 'w').close()
            filewrite = open(pathIterTMP, 'a')
            firstLine = 0
            for line in file:
                if firstLine == 0:
                    firstLine = firstLine + 1
                    filewrite.write(str(line))

                else:
                    firstLine = firstLine + 1
                    newline = line.replace(', ', '-')
                    filewrite.write(str(newline))

            file.close()
            filewrite.close()

        LogParameterList = [
            'N_r',
            'N_a',
            'N_g',
            'N_Pe',
            'N_Dl'

        ]
        binNumberDef = [4, 7, 9, 8, 7]
        # bigData = pd.DataFrame()
        # bigData['N_Pe'] = trainingData['N_Pe']
        # bins = np.logspace(start=bigData['N_Pe'].min(),base=10, stop=bigData['N_Pe'].max(), num=binNumber)
        # print bins
        for parameter in zip(LogParameterList, binNumberDef):
            print parameter
            bigData = pd.DataFrame()
            bigData[parameter[0]] = trainingData[parameter[0]]
            bigData['classification'] = trainingData['classification']

            if parameter[0] == 'N_g': # small numbers mess up pandas cut
                bigData[parameter[0]] = trainingData[parameter[0]]*1e11
                low = math.floor(np.log10(bigData[parameter[0]].min())+.1/100)
                high = float(math.ceil(np.log10(bigData[parameter[0]].max())+.1/100))
                bins = np.logspace(low, high, parameter[1],endpoint=True,base=10)
                expon = bigData[bigData['classification'].isin(['exponential'])]
                expon.columns = [str(parameter[0]), 'exponential']

                exponGrouped = expon.groupby(pd.cut(expon[str(parameter[0])], bins,precision=0)).size().reset_index()
                exponGrouped.columns = [str(parameter[0]), 'exponential']

                nonexpon = bigData[bigData['classification'].isin(['nonexponential'])]
                nonexpon.columns = [str(parameter[0]), 'nonexponential']

                nonexponGrouped = nonexpon.groupby(pd.cut(nonexpon[str(parameter[0])], bins,precision=0)).size().reset_index()
                nonexponGrouped.columns = [str(parameter[0]), 'nonexponential']
                CombinedGrouped = pd.concat([exponGrouped, nonexponGrouped['nonexponential']], axis=1)
                print CombinedGrouped
                # CombinedGrouped = CombinedGrouped.divide(1e10,axis='columns')

            else:
                low = math.floor(np.log10(bigData[parameter[0]].min())+.1/100)
                high = float(math.ceil(np.log10(bigData[parameter[0]].max())+.1/100))
                bins = np.logspace(low, high, parameter[1],endpoint=True,base=10)
                expon = bigData[bigData['classification'].isin(['exponential'])]
                expon.columns = [str(parameter[0]), 'exponential']

                exponGrouped = expon.groupby(pd.cut(expon[str(parameter[0])], bins,precision=0)).size().reset_index()
                exponGrouped.columns = [str(parameter[0]), 'exponential']

                nonexpon = bigData[bigData['classification'].isin(['nonexponential'])]
                nonexpon.columns = [str(parameter[0]), 'nonexponential']

                nonexponGrouped = nonexpon.groupby(pd.cut(nonexpon[str(parameter[0])], bins,precision=0)).size().reset_index()
                nonexponGrouped.columns = [str(parameter[0]), 'nonexponential']
                CombinedGrouped = pd.concat([exponGrouped, nonexponGrouped['nonexponential']], axis=1)

            # print CombinedGrouped
            # print CombinedGrouped
            pathIter = os.path.join('figures', 'histograms', 'tmps', str('tmp' + parameter[0] + 'tmp.csv'))

            file = open(pathIter, "w").close()  # if there is an existing file with this name, clear it.
            file = open(pathIter, "a")
            file.write(str(parameter[0] + ',' + 'exponential,nonexponential\n'))
            import fileinput

            for j in range(0, len(bins) - 1):  # when we get a range, there's always one left so -1
                if j < len(bins) - 1:
                    for i in range(0, 3):
                        if i == 0:
                            a = 0
                            a = CombinedGrouped.ix[j][i][1:-1]
                            file.write(str(a + ','))
                        elif i < 2:
                            b = 0
                            b = CombinedGrouped.ix[j][i]
                            file.write(str(b) + ',')
                        else:
                            b = 0
                            b = CombinedGrouped.ix[j][i]
                            file.write(str(b))
                    file.write('\n')
                else:
                    for i in range(0, 3):
                        if i == 0:
                            a = 0
                            a = CombinedGrouped.ix[j][i][1:-1]
                            file.write(str('' + a))
                        elif i < 2:
                            b = 0
                            b = CombinedGrouped.ix[j][i]
                            file.write(str(b) + ',')
                        else:
                            b = 0
                            b = CombinedGrouped.ix[j][i]
                            file.write(str(b))
                    file.write('\n')
            file.close()

            file = open(pathIter, "r")
            pathIterTMP = os.path.join('figures', 'histograms', str(parameter[0] + '.csv'))
            filewrite = open(pathIterTMP, 'w').close()
            filewrite = open(pathIterTMP, 'a')
            firstLine = 0
            for line in file:
                if firstLine == 0:
                    firstLine = firstLine + 1
                    filewrite.write(str(line))
                else:
                    firstLine = firstLine + 1
                    newline = line.replace(', ', '-')
                    # line = re.sub('"', '', line.rstrip())
                    filewrite.write(str(newline))

            file.close()
            filewrite.close()

        # Looking at diversity of upper and lower branches
