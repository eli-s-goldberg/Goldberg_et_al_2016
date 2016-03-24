"""
This file imports the database and does a series of recursive histograms, where the data can be binned as a function
of one or more variables.
For instance, if the peclet numbers for the experiments range from 90 to  1e6, but I want to know which experiments
within that range result in an exponential profile, then I have to do this.
"""
import os
import math
import numpy as np
import pandas as pd

from helper_functions import (bin_to_rp_shape, make_dirs)

# Default database
TRAINING_PATH = os.path.join('output', 'data', 'training_data.csv')
TARGET_PATH = os.path.join('output', 'data', 'target_data.csv')

_CATEGORICAL_FEATURES_TO_DROP = [
    u'collector_coating=FeOOH', u'collector_coating=IronOxide',
    u'collector_coating=None', u'enm_id=Ag', u'enm_id=C60',
    u'enm_id=CeO2', u'enm_id=CuO', u'enm_id=Fe', u'enm_id=MWCNT',
    u'enm_id=QD', u'enm_id=TiO2', u'enm_id=ZnO', u'enm_id=nBiochar',
    u'enm_id=nHAP', u'nom_id=Alg', u'nom_id=Citric', u'nom_id=FA',
    u'nom_id=Formic', u'nom_id=HA', u'nom_id=None', u'nom_id=Oxalic',
    u'nom_id=SRHA', u'nom_id=TRIZMA']

_PARAMETER_BIN_SIZE_DICT = {
    'n_Lo': 13,
    'm_inf': 26,
    'n_asp': 4,
    'n_z1': 10,
    'n_z2': 11,
    'n_asp_c': 9,
    'influent_concentration_enm': 11,
    'concentration_nom': 10,
    'n_att': 7,
    'n_por': 10,
    'n_g': 9,
    'n_Pe': 8,
    'n_dl': 7
}
_PARAMETER_BIN_SPACE_DICT = {
    'n_Lo': 'linear',
    'm_inf': 'linear',
    'n_asp': 'log',
    'n_z1': 'linear',
    'n_z2': 'linear',
    'n_asp_c': 'linear',
    'influent_concentration_enm': 'linear',
    'concentration_nom': 'linear',
    'n_att': 'log',
    'n_por': 'linear',
    'n_g': 'log',
    'n_Pe': 'log',
    'n_dl': 'log'
}

def main(output_dir='output', training_path=TRAINING_PATH, target_path=TARGET_PATH):
    """Generate histograms."""
    target_data = pd.read_csv(target_path)
    training_data = pd.read_csv(training_path)
    target_data['classification'] = target_data.apply(bin_to_rp_shape, axis=1)

    training_data = training_data.drop(_CATEGORICAL_FEATURES_TO_DROP, 1)
    feature_names = list(training_data.columns.values)

    combined_data = training_data.copy(deep=True)
    combined_data['classification'] = target_data['classification']

    make_dirs(os.path.join(output_dir, 'histograms'))
    for parameter in feature_names:
        output_path = os.path.join(output_dir, 'histograms', str(parameter+'.csv'))
        if _PARAMETER_BIN_SPACE_DICT.get(parameter) == 'linear':
            low = math.floor(combined_data[parameter].min())  # low floor boundary for parameter
            high = math.ceil(combined_data[parameter].max())  # high ceiling boundary for parameter
            bins = np.linspace(low, high, _PARAMETER_BIN_SIZE_DICT.get(parameter), endpoint=True)
            bins = np.unique(bins)  # ensure that bin edges are unique
            exponential_data = combined_data[combined_data['classification'].isin(['exponential'])]
            exponential_data_grouped = exponential_data.groupby(
                pd.cut(exponential_data[parameter],
                       bins, precision=0)).size().reset_index()
            exponential_data_grouped.columns = [parameter, 'exponential']
            nonexponential_data = combined_data[combined_data['classification'].isin(['nonexponential'])]
            nonexponential_data_grouped = nonexponential_data.groupby(
                pd.cut(nonexponential_data[parameter],
                       bins, precision=0)).size().reset_index()
            nonexponential_data_grouped.columns = [parameter, 'nonexponential']
            linear_grouped_data = pd.concat([exponential_data_grouped,
                                             nonexponential_data_grouped['nonexponential']], axis=1)
            linear_grouped_data.to_csv(output_path, index=False)
        else:
            low = math.floor(np.log10(combined_data[parameter].min()))
            high = math.ceil(np.log10(combined_data[parameter].max()))
            bins = np.logspace(low, high,
                               _PARAMETER_BIN_SIZE_DICT.get(parameter), endpoint=True, base=10)
            bins = np.unique(bins)  # ensure that bin edges are unique
            exponential_data = combined_data[combined_data['classification'].isin(['exponential'])]
            exponential_data_grouped = exponential_data.groupby(pd.cut(exponential_data[str(parameter)],
                                                                       bins, precision=0)).size().reset_index()
            exponential_data_grouped.columns = [parameter, 'exponential']
            nonexponential_data = combined_data[combined_data['classification'].isin(['nonexponential'])]
            nonexponential_data_grouped = nonexponential_data.groupby(
                pd.cut(nonexponential_data[parameter], bins, precision=0)).size().reset_index()
            nonexponential_data_grouped.columns = [parameter, 'nonexponential']
            nonlinear_grouped_data = pd.concat([exponential_data_grouped,
                                                nonexponential_data_grouped['nonexponential']], axis=1)
            nonlinear_grouped_data.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()
