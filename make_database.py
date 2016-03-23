"""
 -- Run this file first to generate the database.  --
This script converts many of the dimensioned physical, chemical, and electrical conditions
into dimensionless parameters. Conversions assume a constant temperature (298.15K), with the
associated kinematic viscosity and density of water to match. See 'helper_functions.py' for detail
on specific assumptions for the conversion and references.
"""
import os
import pandas as pd

from helper_functions import (
    binary_rp_class_assign, dim_aspect_ratio_assign, dim_peclet_num_assign,
    attraction_number, gravity_number, debye_length, mass_flow, electrokinetic1,
    return_ionic_strength,electrokinetic2, rel_permittivity, return_valence,
    electrolyte_relative_concentration, sorbed_mass_ratio,london_force,
    porosity_happel, edl_force, column_aspect_ratio, one_hot_dataframe)

# Where is the initial database?
IMPORT_DATABASE_PATH = os.path.join(
    os.path.dirname(__file__), 'transport_database', 'enmTransportData.xlsx')

# Where will converted database go?
EXPORT_TRAINING_PATH = os.path.join(
    os.path.dirname(__file__), 'transport_database', 'training_data.csv')
EXPORT_TARGET_PATH = os.path.join(
    os.path.dirname(__file__), 'transport_database', 'target_data.csv')

def main(path='.', database_path=IMPORT_DATABASE_PATH):
    """Pre-process database."""
    alpha_dataset = pd.read_excel(database_path)

    # Identify columns that are not needed for the assessment and drop
    drop_column_list = [
        'publication_title',
        'author',
        'year',
        'experiment_id',
        'dispersivity',
        'mass_balance_effluent',
        'mass_balance_retained',
        'mass_balance_effluent_normalized',
        'mass_balance_retained_normalized',
        'notes1',
        'notes2',
        'notes3',
        'notes4',
        'notes5']

    alpha_dataset = alpha_dataset.drop(drop_column_list, 1)

    # remove any row without data (NA)
    alpha_dataset = alpha_dataset.dropna()

    # save the dataset for later inspection and use as refinedDataset
    alpha_dataset.to_csv(os.path.join(path, 'refinedDataset.csv'))

    # copy the refined dataset to a new variable.
    training_data = alpha_dataset.copy(deep=True)

    # drop the classification from the training data (that wouldn't be fun:)
    training_data = training_data.drop(['rp_shape'], 1)

    # Set the target data, copy into a new database and binarize
    # inputs to exponential or nonexponential.
    target_data = pd.DataFrame(alpha_dataset.rp_shape)
    target_data['rp_shape'] = target_data.apply(binary_rp_class_assign, axis=1)

    # Apply dimensionless number feature dimension reduction, start by
    # assigning assumed constants (i.e., temp). Note that this requires some
    # data to be factorized. This will be changed in later versions, but now
    # it requires that we factorize, then apply and conform to dimensionless
    # parameters, and then reencode to recover categorical variables.
    # Note that 1) the valence and electrolyte relative concentration need to be
    # calculated before the ionic strength, and 2) the ionic strength needs to be
    # calculated before the debye_length.

    training_data['enm_relative_permittivity'] = training_data.apply(
        rel_permittivity,axis=1)
    training_data['valence'] = training_data.apply(
        return_valence,axis=1)
    training_data['electrolyte_rel_conc'] = training_data.apply(
        electrolyte_relative_concentration,axis=1)
    training_data['ionic_strength'] = training_data.apply(
        return_ionic_strength,axis=1)
    training_data['debye_length'] = training_data.apply(
        debye_length,axis=1)
    training_data['n_dl'] = training_data.apply(
        edl_force,axis=1)
    training_data['n_asp'] = training_data.apply(dim_aspect_ratio_assign, axis=1)
    training_data['n_att'] = training_data.apply(attraction_number, axis=1)
    training_data['n_g'] = training_data.apply(gravity_number, axis=1)*1e9 # very small numbers, must change
    training_data['n_Pe'] = training_data.apply(dim_peclet_num_assign, axis=1)
    training_data['n_Lo'] = training_data.apply(london_force, axis=1)
    training_data['n_por'] = training_data.apply(porosity_happel,axis=1)
    training_data['n_dl'] = training_data.apply(edl_force, axis=1)
    training_data['m_inf'] = training_data.apply(mass_flow, axis=1) *1e6 # # from kg to mg
    # training_data['n_m_sorbed'] = training_data.apply(sorbed_mass_ratio, axis=1)
    training_data['n_z1'] = training_data.apply(electrokinetic1, axis=1)
    training_data['n_z2'] = training_data.apply(electrokinetic2, axis=1)
    training_data['n_asp_c'] = training_data.apply(column_aspect_ratio, axis=1)
    training_data['influent_concentration_enm'] = \
        training_data['influent_concentration_enm']*1e3 # from kg/m^3 to mg/L
    training_data['concentration_nom'] = \
        training_data['concentration_nom']*1e3 # from kg/m^3 to mg/L
    # Output a final copy of the training data for later use
    training_data.to_csv(
        os.path.join(path, 'trainingdataAll.csv'), index=False, header=True)

    # Drop overlapping features - Combination assessment: temporary
    training_data = training_data.drop(
        ['enm_relative_permittivity',
         'valence',
         'electrolyte_rel_conc',
         'electrolyte_id',
         'column_length',
         'column_width',
         'porosity',
         'darcy_velocity',
         'influent_pore_volumes',
         'enm_density',
         'enm_diameter',
         'collector_diameter',
         'electrolyte_concentration',
         'enm_zeta_potential',
         'collector_zeta_potential',
         'ionic_strength',
         'electrolyte_id',
         'ph',
         'porosity',
         'debye_length',
         'enm_isoelectric_point',
         'hamaker_constant_combined'], 1)

    # encode the categorical variables using a one-hot scheme so they're
    # correctly considered by decision tree methods
    training_data, _, _ = one_hot_dataframe(
        training_data, ['enm_id','collector_coating', 'nom_id'], replace=True)

    training_data.to_csv(EXPORT_TRAINING_PATH, index=False)
    target_data.to_csv(EXPORT_TARGET_PATH, index=False)

if __name__ == '__main__':
    main()
