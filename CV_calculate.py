import pandas as pd
import numpy
import scipy
from scipy import stats
import os
import csv, itertools, json

path = '/Users/future/Google Drive/phd_root/programming_root/enmTransportPrediction/'


branch0 = pd.read_csv(os.path.join(path, 'output','training_and_target_data','trainingdata.csv'))
branch0 = branch0.dropna()
branch0 = branch0.drop(['NMId','Coating','TypeNOM'],1)

branchParams0 = list(branch0)
print len(branchParams0)

TrackBranch0_Radar = pd.DataFrame()
for params in branch0:
    # print params
    if params=='N_Z2':
        def addOne(row):
            a = row+1
            return a

        branchParamColumn =  branch0[str(params)]
        branchParamColumn = branchParamColumn.apply(addOne,1)
        # print branchParamColumn

        totalVals = len(branchParamColumn)
        mean = branchParamColumn.mean()
        stdev = branchParamColumn.std()
        CV = scipy.stats.variation(branchParamColumn,axis=0)
    else:
        branchParamColumn =  branch0[str(params)]
        totalVals = len(branchParamColumn)
        mean = branchParamColumn.mean()
        stdev = branchParamColumn.std()
        CV = scipy.stats.variation(branchParamColumn,axis=0)
    radarOut = pd.DataFrame([CV],columns=[str(params)])
    TrackBranch0_Radar=pd.concat([TrackBranch0_Radar,radarOut],1)


branch1 = pd.read_excel(os.path.join(path, 'transport_database', 'upperBranch.xlsx'))
branch1 = branch1.dropna()
branch1 = branch1[branch1.NMId != 'TiO2']
branch1 = branch1.drop(['NMId','Coating','TypeNOM','Classification'],1)

branchParams1 = list(branch1)

TrackBranch1_Radar = pd.DataFrame()
for params in branch1:
    # print params
    if params=='N_Z2':
        def addOne(row):
            a = row+1
            return a

        branchParamColumn =  branch1[str(params)]
        branchParamColumn = branchParamColumn.apply(addOne,1)
        # print branchParamColumn

        totalVals = len(branchParamColumn)
        mean = branchParamColumn.mean()
        stdev = branchParamColumn.std()
        CV = scipy.stats.variation(branchParamColumn,axis=0)
    else:
        branchParamColumn =  branch1[str(params)]
        totalVals = len(branchParamColumn)
        mean = branchParamColumn.mean()
        stdev = branchParamColumn.std()
        CV = scipy.stats.variation(branchParamColumn,axis=0)
    radarOut = pd.DataFrame([CV],columns=[str(params)])
    TrackBranch1_Radar=pd.concat([TrackBranch1_Radar,radarOut],1)


branch2 = pd.read_excel(os.path.join(path, 'transport_database', 'lowerBranch.xlsx'))
branch2 = branch2.dropna()
branch2 = branch2.drop(['NMId','Coating','TypeNOM','Classification'],1)

branchParams2 = list(branch2)

TrackBranch2_Radar = pd.DataFrame()
for params in branch2:
      # print params
    if params=='N_Z2':
        def addOne(row):
            a = row+1
            return a

        branchParamColumn =  branch2[str(params)]
        branchParamColumn = branchParamColumn.apply(addOne,1)

        totalVals = len(branchParamColumn)
        mean = branchParamColumn.mean()
        stdev = branchParamColumn.std()
        CV = scipy.stats.variation(branchParamColumn,axis=0)

    else:
        branchParamColumn =  branch2[str(params)]
        totalVals = len(branchParamColumn)
        mean = branchParamColumn.mean()
        stdev = branchParamColumn.std()
        CV = scipy.stats.variation(branchParamColumn,axis=0)

    radarOut = pd.DataFrame([CV],columns=[str(params)])
    TrackBranch2_Radar=pd.concat([TrackBranch2_Radar,radarOut],1)


branch3 = pd.read_excel(os.path.join(path, 'transport_database', 'upper2.xlsx'))
branch3 = branch3.dropna()
branch3 = branch3.drop(['NMId','Coating','TypeNOM','Classification'],1)

branchParams3 = list(branch2)

TrackBranch3_Radar = pd.DataFrame()
for params in branch3:
      # print params
    if params=='N_Z2':
        def addOne(row):
            a = row+1
            return a

        branchParamColumn =  branch3[str(params)]
        branchParamColumn = branchParamColumn.apply(addOne,1)

        totalVals = len(branchParamColumn)
        mean = branchParamColumn.mean()
        stdev = branchParamColumn.std()
        CV = scipy.stats.variation(branchParamColumn,axis=0)

    else:
        branchParamColumn =  branch3[str(params)]
        totalVals = len(branchParamColumn)
        mean = branchParamColumn.mean()
        stdev = branchParamColumn.std()
        CV = scipy.stats.variation(branchParamColumn,axis=0)

    radarOut = pd.DataFrame([CV],columns=[str(params)])
    TrackBranch3_Radar=pd.concat([TrackBranch3_Radar,radarOut],1)



dfOut = pd.concat([TrackBranch1_Radar,TrackBranch2_Radar,TrackBranch3_Radar],0)

# print dfOut
dfOut = dfOut[['M_inj','ConcIn','ConcHA', 'N_r','N_a','N_g','N_Pe','N_Lo','N_Dl','N_as','N_CA','N_Z2','N_Z1']]
print dfOut.head()
dfOutParams = list(dfOut)
# print dfOut.div(dfOut.max(axis=1),axis=0)

normalizedOut = pd.DataFrame()
normalizedOut['yellow'] = dfOut.iloc[0]/(dfOut.iloc[0])
normalizedOut['red'] = dfOut.iloc[1]/(dfOut.iloc[0])
normalizedOut['blue']= dfOut.iloc[2]/(dfOut.iloc[0])
normalizedOut = normalizedOut.transpose()
dfOut = normalizedOut

pathWrite = os.path.join(path,'figures', 'radarCharts', str('RadardData.txt'))

file = open(pathWrite, "w").close()  # if there is an existing file with this name, clear it.
file = open(pathWrite, "a")

# print dfOutParams
for i in range(0,3):
    print i
    file.write("[\n")
    for params in dfOutParams:
        value = dfOut[str(params)].as_matrix()
        file.write('{axis: "CV for ' + params+ '", value: ' + str(value[i])+'},\n')
    file.write("],")
file.close()