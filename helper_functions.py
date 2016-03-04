"""Helper functions for optimal_tree pipeline."""

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

def binaryRPClassAssign(row):
    if row.ObsRPShape in ["EXP", "HE"]:
        return 0
    else:
        return 1

def binaryRPClassAssignLabels(row):
    if row.ObsRPShape in ["EXP", "HE"]:
        return "exponential"
    else:
        return "nonexponential"

def dimAspectRatioAssign(row):
    return float(row.PartDiam / row.CollecDiam)

def dimPecletNumAssign(row):
    bolzConstant = float(1.3806504 * 10 ** -23)
    tempK = 25 + 273.15
    diffusionCoef = float(bolzConstant * tempK / (3 * math.pi * 8.94e-4 * row.PartDiam))
    return float(row.Darcy * row.CollecDiam / diffusionCoef)


def gravitationalNumber(row):
    p_radius = float((row.PartDiam) / 2)
    bolzConstant = float(1.3806504 * 10 ** -23)
    tempK = 25 + 273.15
    # TODO(peterthenelson) 4/3 is a bug (==> 1, not 1.3...).
    return float((4/3) * math.pi * (p_radius**4) * (row.PartDensity-1000) *
                 9.81 / (bolzConstant * tempK))


def attractionNumber(row):
    p_radius = float((row.PartDiam) / 2)
    denominator = 12 * math.pi * p_radius ** 2 * row.Darcy
    return float(row.Hamaker / denominator)


def gravityNumber(row):
    temp = row.tempKelvin
    abs_visc = float(math.exp(-3.7188 + (578.919 / (-137.546 + temp))))
    p_radius = float((row.PartDiam) / 2)
    numerator = (2 * p_radius ** 2) * (row.PartDensity - 1000) * 9.81
    demoninator = 8 * abs_visc * row.Darcy
    return float(numerator / demoninator)


def debyeLength(row):
    permFreeSpace = float(8.854 * 10 ** -12)
    bolzConstant = float(1.3806504 * 10 ** -23)
    elecCharge = float(1.602176466 * 10 ** -19)
    numAvagadro = float(6.02 * 10 ** 23)
    if row['SaltType'] == 3:
        ionStr1 = float(1 * 10 ** (float(row['pH']) - 14))
        ionStr2 = float(1 * 10 ** (-1 * float(row['pH'])))
        ZiCi = float(1 ** 2 * ionStr1 + 1 ** 2 * ionStr2)
        return float((1 / (float((numAvagadro * elecCharge ** 2 / (
            permFreeSpace * float(row['relPermValue']) * bolzConstant *
            float(row['tempKelvin'])) * ZiCi) ** 0.5))))
    elif row['SaltType'] == 1:
        ZiCi = float(2 ** 2 * float(row['IonStr']) + 1 ** 2 * 2 *
                     float(row['IonStr']))
        return float((1 / (float((numAvagadro * elecCharge ** 2 / (
            permFreeSpace * float(row['relPermValue']) * bolzConstant *
            float(row['tempKelvin'])) * ZiCi) ** 0.5))))
    elif row['IonStr'] == 0:
        return 1000e-9  # about 1um
    else:
        ZiCi = float(1 ** 2 * float(row['IonStr']) + 1 ** 2 *
                     float(row['IonStr']))
        return float((1 / (float((numAvagadro * elecCharge ** 2 / (
            permFreeSpace * float(row['relPermValue']) * bolzConstant *
            float(row['tempKelvin'])) * ZiCi) ** 0.5))))


def massFlow(row):
    L = row.colLength
    W = row.colWidth
    A = float(math.pi / 4 * row.colWidth ** 2)
    p = row.Poros
    PVs = row.PvIn
    return float(L * W * A * p * PVs)


def electrokinetic1(row):
    v = 7.83e-9  # dialectric constant of water at 25C in coulombs/volt/m
    a = row.PartDiam / 2  # particle radius
    Zp = row.PartZeta / 1e3  # particle zeta potential
    Zc = row.CollecZeta / 1e3  # collector zeta potential
    k = float(1.3806504 * 10 ** -23)  # boltzmann constant
    T = row.tempKelvin  # temperature

    return float(v * a * (Zp ** 2 + Zc ** 2) / (4 * k * T))


def electrokinetic2(row):
    Zp = row.PartZeta / 1e3  # particle zeta potential
    Zc = row.CollecZeta / 1e3  # collector zeta potential
    numerator = 2 * (Zp / Zc)
    denominator = 1 + (Zp / Zc) ** 2
    return float(numerator / denominator)

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


def relPermittivity(row):
    return _REL_PERMITTIVITIES.get(row['NMId'], 10.0)


def changEta0(row):
    N_Dl = row.N_Dl
    N_Z1 = row.N_Z1
    N_Z2 = row.N_Z2
    N_Lo = row.N_Lo
    N_as = row.N_as
    N_r = row.N_r
    N_Pe = row.N_Pe
    N_g = row.N_g

    return float(0.024 * N_Dl ** (0.969) * N_Z1 ** (-0.423) * N_Z2 ** (2.880) * N_Lo ** 1.5 + \
                 3.176 * N_as ** (0.333) * N_r ** (-0.081) * N_Pe ** (-0.715) * N_Lo ** (2.687) + \
                 0.222 * N_as * N_r ** (3.041) * N_Pe ** (-0.514) * N_Lo ** (0.125) + \
                 N_r ** (-0.24) * N_g ** (1.11) * N_Lo)


def londonForce(row):
    k = float(1.3806504 * 10 ** -23)
    T = row.tempKelvin
    A = row.Hamaker
    return float(A / (6 * k * T))


def zetaRatioKnockout(row):
    # TODO(peterthenelson) I doubt modifying the row itself is intended.
    if row.N_z < 0:
        row.N_z = 0
    return row.N_z


# TODO(peterthenelson): I think this is unused and can be deleted
# def heldout_score(clf, X_test, y_test):
#    """compute deviance scores on ``X_test`` and ``y_test``. """
#    clf.fit(X_test, y_test)
#    score = np.zeros((n_estimators,), dtype=np.float64)
#    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
#        score[i] = clf.loss_(y_test, y_pred)
#    return score


def porosityHappel(row):
    p = float(row.Poros)
    gam = (1 - p) ** (.333333333)
    numerator = 2 * (1 - gam ** 5)
    denominator = 2 - 3 * gam + 3 * gam ** 5 - 2 * gam ** 6
    return float(numerator / denominator)


def debyeNumber(row):
    D = row.D_l
    dp = row.PartDiam
    return float(dp / D)

def rulesGradientBoost(clf, features, labels, node_index=0):

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
        samplesTerminal = clf.tree_.n_node_samples[node_index]

        impurity = round(clf.tree_.impurity[node_index], 3)

        node['name'] = ', '.join(('{} {}'.format('%.2f' % Decimal(count), label)
                                  for count, label in count_labels))

        node['impurity'] = "{}".format(impurity)
        node['samples'] = "{}".format(samplesTerminal)
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

def NMIDClassAssign(row):
    return _NMID_CLASSES[row.NMId]

_SALT_CLASSES = [
    "NaCl",   # 0
    "CaCl2",  # 1
    "KCl",    # 2
    "None",   # 3
    "KNO3",   # 4
    "NaHCO3", # 5
]

def SaltClassAssign(row):
    return _SALT_CLASSES[row.SaltType]

_COATING_CLASSES = ["None", "IronOxide", "FeOOH"]

def CoatingClassAssign(row):
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

def TypeNOMClassAssign(row):
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
