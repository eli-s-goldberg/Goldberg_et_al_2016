# coding=utf-8
"""Helper functions for optimal_tree pipeline."""

from __future__ import division
import ast
from decimal import Decimal
import errno
import math
import os
import pandas
from sklearn.feature_extraction import DictVectorizer

def make_dirs(path):
    """Recursively make directories, ignoring when they already exist.

    See: http://stackoverflow.com/a/600612/1275412
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST or not os.path.isdir(path):
            raise

def _str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]

def add_boolean_argument(parser, name, default=False):
    """Add a boolean argument to an ArgumentParser instance."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--' + name, nargs='?', default=default, const=True, type=_str_to_bool)
    group.add_argument('--no' + name, dest=name, action='store_false')

# Constants
_BOLTZ = 1.3806504e-23  # # Boltzmann's constant [J/K]
_TEMP_K = 25 + 273.15  # Assumed all experiments at 25 C
_GRAV = 9.81  # Gravitational constant [m/s^2]
_PERM_FREE_SPACE = 8.854e-12  # Permittivity of vacuum. [C^2/(N.m^2)]
_ELEC_CHARGE = 1.602176466e-19  # Fundamental charge [C].
_AVOGADRO = 6.022140857e23  # Avogadro's number [1/mol]
_WATER_DIALECTRIC = 80.4 * _PERM_FREE_SPACE  # Dialectric constant of water at 25C in coulombs/volt/m.
_WATER_DENSITY = 1000 * 0.9970479  # Density of water at 25C
_KINEMATIC_WATER_VISCOSITY = 8.94e-4  # Kinematic viscosity of water at 25C in Pa.s
_COLLECTOR_DENSITY = 2650  # Density of quartz sand, used as a proxy for collector [kg/m^3]

# Exponential response profile shape names.
_EXP_RP_SHAPES = ["EXP", "HE"]

# Dictionaries
# -- Electrolyte names and valences
_ELECTROLYTE_VALENCE = {
    'NaCl': [1, -1],
    'CaCl2': [2, -1],
    'KCl': [1, -1],
    'KNO3': [1, -1],
    'NaHCO3': [1, -1],
    'H3PO4': [1, -3],
    'Na3PO4': [1, -1],
    'none': [0, 0]}
# -- Electrolyte names and relative concentrations of subcomponents
_ELECTROLYTE_CONCENTRATION = {
    'NaCl': [1, 1],
    'CaCl2': [1, 2],
    'KCl': [1, 1],
    'KNO3': [1, 1],
    'NaHCO3': [1, 1],
    'H3PO4': [3, 1],
    'Na3PO4': [3, 1],
    'none': [0, 0]}
# -- ENM names and relative permittivity
_REL_PERMITTIVITIES = {
    'C60': 4.4,  # C60
    'TiO2': 110,  # TiO2
    'ZnO': 2,  # ZnO
    'CuO': 18.1,  # CuO
    'MWCNTs': 1328,  # MWCNTs
    'Ag': 2.65,  # Ag
    'CeO2': 26,  # CeO2
    'Iron Oxide': 14.2,  # Iron Oxide
    # 8: 3.9,  # SiO2 removed because no SiO2 with hamakers
    # Originally reported at 7, but found it at 15.4 at 100Hz in Preparation and
    # dielectric property of sintered monoclinic hydroxyapatite.
    'nHAP': 15.4,  # nHAP
    # Toshiyuki Ikoma, Atsushi Yamazaki, Satoshi Nakamura, Masaru Akao
    # nBiochar Dielectric properties and microwave heating of oil palm biomass
    # and biochar
    'nBiochar': 2.9,  # biochar
    # http://scholar.lib.vt.edu/theses/available/etd-04262005-181042/unrestricted/Ch2Theory.pdf
    'QDs': 10.0,  # CdSe
}
_BINARY_TO_RP_SHAPE_DIC = {
    1: "nonexponential",
    0: "exponential"
}

def electrokinetic1(row):
    """
    notes:  1) zeta potentials are in mV. if in V, remove the 1e3
            2) relative dialectric is for water, if this is not true,
            make a column and change the function.
    references:
            (1) You-Im Chang and Hsun-Chih Chan.
            "Correlation equation for predicting filter coefficient under
            unfavorable deposition conditions".
            AIChE journal, 54(5):1235–1253, 2008.
            (2) Rajagopalan, R. & Kim, J. S.
            "Adsorption of brownian particles in the presence of potential
            barriers: effect of different modes of double-layer interaction".
            Journal of Colloid and Interface Science 83, 428–448 (1981).
    :param row:
    :return: 1st electrokinetic parameter
    """
    a = row.enm_diameter / 2  # particle radius
    zp = row.enm_zeta_potential / 1e3  # particle zeta potential in V
    zc = row.collector_zeta_potential / 1e3  # collector zeta potential
    return _WATER_DIALECTRIC * a * (zp ** 2 + zc ** 2) / (4 * _BOLTZ * _TEMP_K)

def electrokinetic2(row):
    """
    notes:  1) zeta potentials are in mV. if in V, remove the 1e3
            2) relative dialectric is for water, if this is not true,
            make a column and change the function.
    references:
            (1) You-Im Chang and Hsun-Chih Chan.
            "Correlation equation for predicting filter coefficient under
            unfavorable deposition conditions".
            AIChE journal, 54(5):1235–1253, 2008.

            (2) Rajagopalan, R. & Kim, J. S.
            "Adsorption of brownian particles in the presence of potential
            barriers: effect of different modes of double-layer interaction".
            Journal of Colloid and Interface Science 83, 428–448 (1981).
    :param row:
    :return: 2nd electrokinetic parameter
    """

    zp = row.enm_zeta_potential / 1e3  # particle zeta potential
    zc = row.collector_zeta_potential / 1e3  # collector zeta potential
    numerator = 2 * (zp / zc)
    denominator = 1 + (zp / zc) ** 2
    return numerator / denominator

def return_valence(row):
    """Find valence.

    notes: match ENM and return valence
    :param row:
    :return: corresponding electrolyte valence
    """
    return _ELECTROLYTE_VALENCE.get(row.electrolyte_id, [0, 0])

def electrolyte_relative_concentration(row):
    """Find relative concentration of electrolyte.

    notes: match ENM and return relative concentration of ions
    :param row:
    :return: corresponding relative ion concentration
    """
    return _ELECTROLYTE_CONCENTRATION.get(row.electrolyte_id, [0, 0])

def return_ionic_strength(row):
    """Find ionic strength.

    notes: if the is an electrolyte in solution (e.g., electrolyte_rel_conc != 0),
    calculate ionic strength if not, then use the pH to determine the concentration of ions in solution.
    :param  row:
            valence:
            electrolyte_rel_conc:
    :return:
    """
    if row.electrolyte_concentration != 0:
        c_i_0 = row.electrolyte_concentration * row.electrolyte_rel_conc[0]
        c_i_1 = row.electrolyte_concentration * row.electrolyte_rel_conc[1]
        return 0.5 * (c_i_0 * row.valence[0] ** 2 + c_i_1 * row.valence[1] ** 2)
    else:
        h_plus_ions = 1.0 * 10 ** (row.ph - 14)
        h_plus_valence = 1
        oh_minus_ions = 1.0 * 10 ** (-1 * row.ph)
        oh_minus_valence = 1
        return 0.5 * (h_plus_ions * h_plus_valence ** 2) + (oh_minus_ions * oh_minus_valence ** 2)

def debye_length(row):
    """Calculate Debye length."""
    numerator = _PERM_FREE_SPACE * row.enm_relative_permittivity * _BOLTZ * _TEMP_K
    denominator = 2.0 * _AVOGADRO * _ELEC_CHARGE ** 2.0 * row.ionic_strength
    return (numerator / denominator) ** 0.5

def edl_force(row):
    """Find EDL force.

    references:
            (1) Nathalie Tufenkji and Menachem Elimelech.
            "Correlation Equation for Predicting Single-Collector
            Efficiency in Physicochemical Filtration in Saturated
            Porous Media."
            Environmental Science & Technology, 38(2):529–536, January 2004
    :param row:
    :return: dimensionless EDL force parameter
    """
    return row.enm_diameter / row.debye_length

def dim_aspect_ratio_assign(row):
    """Aspect dimensionless aspect ratio.

    references:
            (1) Nathalie Tufenkji and Menachem Elimelech.
            "Correlation Equation for Predicting Single-Collector
            Efficiency in Physicochemical Filtration in Saturated
            Porous Media."
            Environmental Science & Technology, 38(2):529–536, January 2004
    :param row:
    :return: dimensionless aspect ratio
    """
    return row.enm_diameter / row.collector_diameter

def dim_peclet_num_assign(row):
    """Find Pectle number.

    notes: 1) Assumes temperature of experiment is at 25C.
    references:
            (1) Nathalie Tufenkji and Menachem Elimelech.
            "Correlation Equation for Predicting Single-Collector
            Efficiency in Physicochemical Filtration in Saturated
            Porous Media."
            Environmental Science & Technology, 38(2):529–536, January 2004
    :param row:
    :return: dimensionless peclet number
    """
    stokes_einstein_diffusion = _BOLTZ * _TEMP_K / (3 * math.pi * _KINEMATIC_WATER_VISCOSITY *
                                                    row.enm_diameter)
    return row.darcy_velocity * row.collector_diameter / stokes_einstein_diffusion

def attraction_number(row):
    """Find attraction number.

    notes:  1) Assumes temperature of experiment is at 25C.
    :param row:
    :return:
    """
    denominator = 3.0 * math.pi * _KINEMATIC_WATER_VISCOSITY * row.enm_diameter ** 2 * row.darcy_velocity
    return row.hamaker_constant_combined / denominator

def london_force(row):
    """Find London force.

    notes:  1) Assumes temperature of experiment is at 25C.
    references:
            (1) Nathalie Tufenkji and Menachem Elimelech.
            "Correlation Equation for Predicting Single-Collector
            Efficiency in Physicochemical Filtration in Saturated
            Porous Media."
            Environmental Science & Technology, 38(2):529–536, January 2004
    :param row:
    :return:
    """
    return row.hamaker_constant_combined / (6 * _BOLTZ * _TEMP_K)

def gravity_number(row):
    """
    notes:  1) Assumes temperature of experiment is at 25C.
    references:
            (1) Nathalie Tufenkji and Menachem Elimelech.
            "Correlation Equation for Predicting Single-Collector
            Efficiency in Physicochemical Filtration in Saturated
            Porous Media."
            Environmental Science & Technology, 38(2):529–536, January 2004
    :param row:
    :return:
    """
    p_radius = row.enm_diameter / 2.0
    numerator = (2.0 * p_radius ** 2) * (row.enm_density - _WATER_DENSITY) * _GRAV
    demoninator = 9 * _KINEMATIC_WATER_VISCOSITY * row.darcy_velocity
    return numerator / demoninator

def porosity_happel(row):
    """

    :param row:
    :return:
    """
    gam = (1 - row.porosity) ** (1 / 3)
    numerator = 2 * (1 - gam ** 5)
    denominator = 2 - 3 * gam + 3 * gam ** 5 - 2 * gam ** 6
    return numerator / denominator

def mass_flow(row):
    """Calculate the mass flow."""
    l = row.column_length
    w = row.column_width
    a = math.pi / 4 * row.column_width ** 2
    p = row.porosity
    p_vs = row.influent_pore_volumes
    return l * w * a * p * p_vs

def sorbed_mass_ratio(row):
    """Find ratio of sorbed mass."""
    l = row.column_length
    w = row.column_width
    a = math.pi / 4 * row.column_width ** 2
    p = row.porosity

    total_collector_mass = l * a * (1 - p) * _COLLECTOR_DENSITY

    sorb_ratio = row.m_inj / total_collector_mass
    return sorb_ratio

def column_aspect_ratio(row):
    """Find aspect ratio of column."""
    return row.column_length / row.column_width

def binary_rp_class_assign(row):
    """Translate RP class into binary."""
    if row.rp_shape in _EXP_RP_SHAPES:
        return 0
    else:
        return 1

def binary_rp_class_assign_labels(row):
    """Assign labels based on RP class."""
    if row.rp_shape in _EXP_RP_SHAPES:
        return "exponential"
    else:
        return "nonexponential"

def rel_permittivity(row):
    """Calculate relative permittivity."""
    return _REL_PERMITTIVITIES.get(row.enm_id, 10.0)

def tufenkji_eta0(row):
    """
    notes: the empirical constants are derived empirically for use with CFT.
    references:
            (1) Nathalie Tufenkji and Menachem Elimelech.
            "Correlation Equation for Predicting Single-Collector
            Efficiency in Physicochemical Filtration in Saturated
            Porous Media."
            Environmental Science & Technology, 38(2):529–536, January 2004
    :param row:
    :return: returns the theoretical single collector efficiency
    """
    n_dl = row.N_Dl
    n_z1 = row.N_Z1
    n_z2 = row.N_Z2
    n_lo = row.N_Lo
    n_as = row.N_as
    n_r = row.N_r
    n_pe = row.N_Pe
    n_g = row.N_g

    return (0.024 * n_dl ** (0.969) * n_z1 ** (-0.423) * n_z2 ** (2.880) * n_lo ** 1.5 +
            3.176 * n_as ** (0.333) * n_r ** (-0.081) * n_pe ** (-0.715) * n_lo ** (2.687) +
            0.222 * n_as * n_r ** (3.041) * n_pe ** (-0.514) * n_lo ** (0.125) +
            n_r ** (-0.24) * n_g ** (1.11) * n_lo)

def rules_gradient_boost(clf, features, labels, node_index=0):
    """Structure of rules in a fit decision tree classifier

    Parameters
    ----------
    clf : DecisionTreeClassifier
        A tree that has already been fit.

    features, labels : lists of str
        The names of the features and labels, respectively.

    """
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        samples = clf.tree_.n_node_samples[node_index]
        impurity = round(clf.tree_.impurity[node_index], 2)

        # TODO(peterthenelson) Is it intended to ignore the label in output?
        node['name'] = ', '.join('{} '.format(count)
                                 for count, _ in count_labels)
        if Decimal(node['name']) < 0:
            print ast.literal_eval(node['name'])

            node['name'] = '{}, nonexponential'.format(
                round(Decimal(node['name']), 2))  # 'nonexponential'

        else:
            # node['name'] = 'exponential'
            node['name'] = '{}, exponential'.format(
                round(Decimal(node['name']), 2))
        node['samples'] = samples
        node['impurity'] = "{}".format(impurity)
        # node['samples'] = ', '.join('{}'.format(samples))


    else:
        feature = features[clf.tree_.feature[node_index]]
        threshold = '%.3E' % Decimal(clf.tree_.threshold[node_index])
        samples = clf.tree_.n_node_samples[node_index]
        # impurity = clf.tree_.impurity[node_index]


        node['name'] = "{} >= {}".format(feature, threshold)
        node['samples'] = '{}'.format(samples)
        # node['impurity'] = '{}%'.format(impurity)


        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        node['children'] = [rules(clf, features, labels, right_index),
                            rules(clf, features, labels, left_index)]
    # print node
    return node

def rules(clf, features, labels, node_index=0):
    """Structure of rules in a fit decision tree classifier.

    Parameters
    ----------
    clf : DecisionTreeClassifier
        A tree that has already been fit.

    features, labels : lists of str
        The names of the features and labels, respectively.

    """
    node = {}

    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        samples_terminal = clf.tree_.n_node_samples[node_index]

        impurity = round(clf.tree_.impurity[node_index], 3)

        node['name'] = ', '.join(('{} {}'.format('%.2f' % Decimal(count), label)
                                  for count, label in count_labels))

        node['impurity'] = "{}".format(impurity)
        node['samples'] = "{}".format(samples_terminal)
        # print node['samples']

    else:
        samples = clf.tree_.n_node_samples[node_index]
        feature = features[clf.tree_.feature[node_index]]
        threshold = '%.3E' % Decimal(clf.tree_.threshold[node_index])

        node['name'] = "{} >= {}".format(feature, threshold)
        node['samples'] = '{}'.format(samples)

        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        node['children'] = [rules(clf, features, labels, right_index),
                            rules(clf, features, labels, left_index)]
    return node

def one_hot_dataframe(data, cols, replace=False):
    """Do hot encoding of categorical columns in a pandas DataFrame.

    See:
    http://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing
    .OneHotEncoder
    http://scikit-learn.org/dev/modules/generated/sklearn.feature_extraction.DictVectorizer.html

    https://gist.github.com/kljensen/5452382

    Takes a dataframe and a list of columns that need to be encoded.
        Returns a 3-tuple comprising the data, the vectorized data,
        and the fitted vectorizor.
    """
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vec_data = pandas.DataFrame(vec.fit_transform(
        data[cols].apply(mkdict, axis=1)).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vec_data)
    return (data, vec_data, vec)

def bin_to_rp_shape(row):
    """Translate binary back to RP shape.

    :param row:
    :return:
    """
    return _BINARY_TO_RP_SHAPE_DIC.get(row.rp_shape, "nonexponential")
