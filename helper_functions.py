"""Helper functions for optimal_tree pipeline."""

from __future__ import division
import ast
from decimal import Decimal
import errno
import math
import os
import pandas
from sklearn.feature_extraction import DictVectorizer

# TODO(peterthenelson) Add docstrings.
# pylint: disable=missing-docstring

def make_dirs(path):
    """Recursively make directories, ignoring when they already exist.

    See: http://stackoverflow.com/a/600612/1275412
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST or not os.path.isdir(path):
            raise

# Exponential response profile shape names.
_EXP_RP_SHAPES = ["EXP", "HE"]
# Boltzmann's constant.
_BOLTZ = 1.3806504e-23
# 25 Celcius in Kelvin.
_TEMP_K = 25 + 273.15
# Gravitational constant.
_GRAV = 9.81
# Permittivity of free space.
_PERM_FREE_SPACE = 8.854e-12
# Fundamental charge in coloumbs.
_ELEC_CHARGE = 1.602176466e-19
# Avogadro's number
_AVOGADRO = 6.02e23
# Dialectric constant of water at 25C in coulombs/volt/m.
_WATER_DIALECTRIC = 7.83e-9

def binary_rp_class_assign(row):
    if row.ObsRPShape in _EXP_RP_SHAPES:
        return 0
    else:
        return 1

def binary_rp_class_assign_labels(row):
    if row.ObsRPShape in _EXP_RP_SHAPES:
        return "exponential"
    else:
        return "nonexponential"

def dim_aspect_ratio_assign(row):
    return row.PartDiam / row.CollecDiam

def dim_peclet_num_assign(row):
    # TODO(peterthenelson) What is this other constant?
    diffusion_coef = _BOLTZ * _TEMP_K / (3 * math.pi * 8.94e-4 * row.PartDiam)
    return row.Darcy * row.CollecDiam / diffusion_coef

def attraction_number(row):
    p_radius = row.PartDiam / 2.0
    denominator = 12 * math.pi * p_radius ** 2 * row.Darcy
    return row.Hamaker / denominator

def gravity_number(row):
    temp = row.tempKelvin
    # TODO(peterthenelson) What are these constants?
    abs_visc = math.exp(-3.7188 + (578.919 / (-137.546 + temp)))
    p_radius = row.PartDiam / 2.0
    numerator = (2 * p_radius ** 2) * (row.PartDensity - 1000) * _GRAV
    demoninator = 8 * abs_visc * row.Darcy
    return numerator / demoninator

def debye_length(row):
    # Calculate zi_ci or return early.
    if row.SaltType == 3:
        ion_str1 = 10 ** (row.pH - 14)
        ion_str2 = 10 ** (-1 * row.pH)
        zi_ci = 1 ** (2 * ion_str1) + 1 ** (2 * ion_str2)
    elif row.SaltType == 1:
        # TODO(peterthenelson) Are there really supposed to be two 2's?
        zi_ci = 2 ** (2 * row.IonStr) + 1 ** (2 * 2 * row.IonStr)
    elif row.IonStr == 0:
        # TODO(peterthenelson) Explain this default and why it comes in this
        # order. Changing the order gives different results.
        return 1000e-9  # about 1um
    else:
        zi_ci = 1 ** (2 * row.IonStr) + 1 ** (2 * row.IonStr)
    # TODO(peterthenelson) Hard to read / understand
    return 1 / ((_AVOGADRO * _ELEC_CHARGE ** 2 / (_PERM_FREE_SPACE *
        row.relPermValue * _BOLTZ * row.tempKelvin) * zi_ci) ** 0.5)

def mass_flow(row):
    l = row.colLength
    w = row.colWidth
    a = math.pi / 4 * row.colWidth ** 2
    p = row.Poros
    p_vs = row.PvIn
    return l * w * a * p * p_vs

def electrokinetic1(row):
    a = row.PartDiam / 2  # particle radius
    zp = row.PartZeta / 1e3  # particle zeta potential
    zc = row.CollecZeta / 1e3  # collector zeta potential
    t = row.tempKelvin  # temperature
    return _WATER_DIALECTRIC * a * (zp ** 2 + zc ** 2) / (4 * _BOLTZ * t)

def electrokinetic2(row):
    zp = row.PartZeta / 1e3  # particle zeta potential
    zc = row.CollecZeta / 1e3  # collector zeta potential
    numerator = 2 * (zp / zc)
    denominator = 1 + (zp / zc) ** 2
    return numerator / denominator

_REL_PERMITTIVITIES = {
    0: 4.4,    # C60
    1: 110.0,  # TiO2
    2: 2.0,    # ZnO
    3: 18.1,   # CuO
    4: 1328.0, # MWCNTs
    5: 2.65,   # Ag
    6: 26.0,   # CeO2
    7: 14.2,   # Iron Oxide
    # 8: 3.9,  # SiO2 removed because no SiO2 with hamakers
    # Originally reported at 7, but found it at 15.4 at 100Hz in Preparation and
    # dielectric property of sintered monoclinic hydroxyapatite.
    8: 15.4,   # nHAP
    # Toshiyuki Ikoma, Atsushi Yamazaki, Satoshi Nakamura, Masaru Akao
    # nBiochar Dielectric properties and microwave heating of oil palm biomass
    # and biochar
    9: 2.9,    # biochar
    # http://scholar.lib.vt.edu/theses/available/etd-04262005-181042/unrestricted/Ch2Theory.pdf
    10: 10.0,  # CdSe
}

def rel_permittivity(row):
    return _REL_PERMITTIVITIES.get(row.NMId, 10.0)

def chang_eta0(row):
    n_dl = row.N_Dl
    n_z1 = row.N_Z1
    n_z2 = row.N_Z2
    n_lo = row.N_Lo
    n_as = row.N_as
    n_r = row.N_r
    n_pe = row.N_Pe
    n_g = row.N_g

    # TODO(peterthenelson) Where do all these constants come from?
    return (0.024 * n_dl ** (0.969) * n_z1 ** (-0.423) * n_z2 ** (2.880) * n_lo ** 1.5 +
            3.176 * n_as ** (0.333) * n_r ** (-0.081) * n_pe ** (-0.715) * n_lo ** (2.687) +
            0.222 * n_as * n_r ** (3.041) * n_pe ** (-0.514) * n_lo ** (0.125) +
            n_r ** (-0.24) * n_g ** (1.11) * n_lo)

def london_force(row):
    return row.Hamaker / (6 * _BOLTZ * row.tempKelvin)

def zeta_ratio_knockout(row):
    if row.N_z < 0:
        return 0.0
    return row.N_z

def porosity_happel(row):
    gam = (1 - row.Poros) ** (1/3)
    numerator = 2 * (1 - gam ** 5)
    denominator = 2 - 3 * gam + 3 * gam ** 5 - 2 * gam ** 6
    return numerator / denominator

def debye_number(row):
    return row.PartDiam / row.D_l

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
        if  Decimal(node['name']) < 0:
            print ast.literal_eval(node['name'])

            node['name'] = '{}, nonexponential'.format(
                round(Decimal(node['name']), 2)) #'nonexponential'

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

_NMID_CLASSES = [
    "C60",     # 0
    "TiO2",    # 1
    "ZnO",     # 2
    "CuO",     # 3
    "MWCNT",   # 4
    "Ag",      # 5
    "CeO",     # 6
    "FeOx",    # 7
    "HAP",     # 8
    "Biochar", # 9
    "QD",      # 10
]

def nmid_class_assign(row):
    return _NMID_CLASSES[row.NMId]

_SALT_CLASSES = [
    "NaCl",   # 0
    "CaCl2",  # 1
    "KCl",    # 2
    "None",   # 3
    "KNO3",   # 4
    "NaHCO3", # 5
]

def salt_class_assign(row):
    return _SALT_CLASSES[row.SaltType]

_COATING_CLASSES = ["None", "IronOxide", "FeOOH"]

def coating_class_assign(row):
    return _COATING_CLASSES[row.Coating]

_NOM_CLASSES = [
    "None",   # 0
    "SRHA",   # 1
    "Alg",    # 2
    "TRIZMA", # 3
    "FA",     # 4
    "HA",     # 5
    "Citric", # 6
    "Oxalic", # 7
    "Formic", # 8
]

def type_nom_class_assign(row):
    return _NOM_CLASSES[row.TypeNOM]

def one_hot_dataframe(data, cols, replace=False):
    """
    Small script that shows hot to do one hot encoding
    of categorical columns in a pandas DataFrame.
    See:
    http://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder
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
