import pandas as pd
from scipy.stats import linregress, wilcoxon
from numpy import log10
import numpy as np
import itertools as it
from numpy.random import normal
from scipy.stats import mode
from scipy.constants import pi

NUM_BOOTSTRAPS = 10000
CONFIDENCE = 99

### Functions


def logRcCalc(logXs, logNs, Umax, Tl, TUmax, g=pi, **kw):
    tn = (10**logXs / (g * 10**logNs * Umax**2))**(1 / 2)
    return log10((Tl - TUmax) / tn)


def JZCA(log_viscosity_Tl, Tl, **kw):
    return log_viscosity_Tl - 2 * log10(Tl)


def Tgr(Tg, Tl, **kw):
    return Tg / Tl


def DTgr(Tg, Tl, Tx, **kw):
    return (Tx - Tg) / (Tl - Tg)


def DTx(Tg, Tx, **kw):
    return Tx - Tg


def DTc(Tg, Tc, **kw):
    return Tc - Tg


def DTl(Tl, Tx, **kw):
    return Tl - Tx


def DTg(Tg, Tl, **kw):
    return Tl - Tg


def Gp(Tg, Tl, Tx, **kw):
    return (Tx - Tg) * Tg / (Tl - Tx)**2


def KH_Tc(Tg, Tl, Tc, **kw):
    return (Tc - Tg) / (Tl - Tc)


def KH_Tx(Tg, Tl, Tx, **kw):
    return (Tx - Tg) / (Tl - Tx)


def KS_Tc(Tg, Tx, Tc, **kw):
    return (Tc - Tx) * (Tc - Tg) / Tg


def KS_Tx(Tg, Tx, Tc, **kw):
    return (Tc - Tx) * (Tx - Tg) / Tg


def Kcr(Tg, Tl, Tx, **kw):
    return (Tl - Tx) / (Tl - Tg)


def Km(Tg, Tx, **kw):
    return (Tx - Tg)**2 / Tg


def Kw_Tx(Tg, Tl, Tx, **kw):
    return (Tx - Tg) / Tl


def Kw_Tc(Tg, Tl, Tc, **kw):
    return (Tc - Tg) / Tl


def Kw2(Tg, Tl, Tx, Tc, **kw):
    return (Tx - Tg) * (Tc - Tx) / Tl


def KLL_Tx(Tg, Tl, Tx, **kw):
    return Tx / (Tg + Tl)


def KLL_Tc(Tg, Tl, Tc, **kw):
    return Tc / (Tg + Tl)


def alpha_Tc(Tl, Tc, **kw):
    return Tc / Tl


def alpha_Tx(Tl, Tx, **kw):
    return Tx / Tl


def beta(Tg, Tl, Tx, **kw):
    return Tx / Tg + Tg / Tl


def beta1(Tg, Tl, Tx, **kw):
    return Tx * Tg / (Tl - Tx)**2


def beta2(Tg, Tl, Tx, **kw):
    return Tg / Tx - Tg / (1.3 * Tl)


def gammac(Tg, Tl, Tx, **kw):
    return (3 * Tx - 2 * Tg) / Tl


def gammam(Tg, Tl, Tx, **kw):
    return (2 * Tx - Tg) / Tl


def delta(Tg, Tl, Tx, **kw):
    return Tx / (Tl - Tg)


def theta(Tg, Tl, Tx, **kw):
    return (Tx + Tg) / Tl * (((Tx - Tg) / Tl)**0.0728)


def xi(Tg, Tl, Tx, **kw):
    return Tg / Tl + (Tx - Tg) / Tx


def phi(Tg, Tl, Tx, **kw):
    return (Tg / Tl) * ((Tx - Tg) / Tg)**0.143


def omegaJAC(Tg, Tl, Tx, **kw):
    return (Tg / Tx) - 2 * Tg / (Tg + Tl)


def omegaji(Tl, Tx, **kw):
    return (Tl * (Tl + Tx)) / (Tx * (Tl - Tx))


def omegaMSEA(Tg, Tl, Tx, **kw):
    return (Tg / (Tx - 2 * Tg)) / (Tg + Tl)


def omega2JNCS(Tg, Tl, Tx, **kw):
    return Tg / (2 * Tx - Tg) - Tg / Tl


def Hdash_Tx(Tg, Tx, **kw):
    return (Tx - Tg) / Tg


def Hdash_Tc(Tg, Tc, **kw):
    return (Tc - Tg) / Tg


function_dic = {
    'KH_Tc': KH_Tc,
    'KH_Tx': KH_Tx,
    'KS_Tc': KS_Tc,
    'KS_Tx': KS_Tx,
    'Kw_Tc': Kw_Tc,
    'Kw_Tx': Kw_Tx,
    'Kw2': Kw2,
    'KLL_Tc': KLL_Tc,
    'KLL_Tx': KLL_Tx,
    'Kcr': Kcr,
    'Km': Km,
    'Gp': Gp,
    'Tgr': Tgr,
    'DTgr': DTgr,
    'DTx': DTx,
    'DTl': DTl,
    'DTg': DTg,
    'DTc': DTc,
    'alpha_Tc': alpha_Tc,
    'alpha_Tx': alpha_Tx,
    'beta': beta,
    'beta1': beta1,
    'beta2': beta2,
    'gammam': gammam,
    'gammac': gammac,
    'delta': delta,
    'omegaJAC': omegaJAC,
    'omegaji': omegaji,
    'omega2JNCS': omega2JNCS,
    'omegaMSEA': omegaMSEA,
    'phi': phi,
    'theta': theta,
    'Hdash_Tc': Hdash_Tc,
    'Hdash_Tx': Hdash_Tx,
    'xi': xi,
    'JZCA': JZCA,
}

GS_names = list(function_dic.keys())

### Noise

log_visc_abs = 0.1

std_pct = {
    'Umax': 2,
}

std_abs = {
    'TUmax': 8,
    'Tx': 8,
    'Tc': 8,
    'Tg': 5,
    'Tl': 5,
    'logNs': 1,
    'logXs': 0,
}

### Data

table_columns = [
    'composition', 'glass', 'Tg', 'Tx', 'Tc', 'Tl', 'logXs', 'logNs', 'Umax',
    'TUmax', 'ninf', 'A', 'T0'
]

table_values = [
    [
        'Li2O.2B2O3', 'LB2', 764, 821, 830, 1190, -2, 3, 0.0029, 1114.0,
        -3.7365, 1684.85, 657.32
    ],
    [
        'Na2O.2B2O3[24]', 'NB2', 730, 813, 825, 1016, -2, 3, 3.34e-05,
        972.008609986176, -3.02715, 1278.98976, 677.09767
    ],
    [
        'SrO.2B2O3 [this work]', 'SB2', 900, 1004, 1023, 1264, -2, 3, 0.00016,
        1215.31644289135, -3.48172, 1614.07463, 817.68718
    ],
    [
        'BaO.2B2O3 [this work]', 'BB2', 866, 972, 988, 1181, -2, 3, 4.31e-05,
        1082.84319567082, -5.0059, 2815.99668, 685.99108
    ],
    [
        'PbO.2B2O3[25]', 'PB2', 710, 826, 866, 1047, -2, 3, 2.1e-06,
        978.775503941144, -3.7365, 1684.85, 657.32
    ],
    [
        'GeO2', 'G', 819, 1090, 1174, 1388, -2, 3, 9.33e-08, 1326.73446866838,
        -7.20869, 17516.2, 48.899
    ],
    [
        'PbO.SiO2', 'PS', 677, 859, 901, 1037, -2, 3, 5.11e-07,
        935.795006629785, -2.689, 1898.5, 555.97
    ],
    [
        'Li2O.2SiO2', 'LS2', 740, 819, 886, 1303, -2, 3, 6.87e-05,
        1209.96784736595, -2.40173, 3082.43, 509.63
    ],
    [
        'Na2O.2SiO2', 'NS2', 713, 897, 918, 1148, -2, 3, 9.92e-07,
        1086.97886018661, -2.99772, 4254.47, 429.2986
    ],
    [
        'MgO.Al2O3.2SiO2', 'MAS2', 1072, 1203, 1238, 1740, -2, 3, 9.05e-06,
        1532.86843611992, -3.97, 5316.0, 762.0
    ],
    [
        'CaO.Al2O3.2SiO2', 'CAS2', 1127, 1280, 1301, 1833, -2, 3, 0.000148,
        1664.1880425871, -3.3163, 3939.07, 866.89
    ],
    [
        'CaO.MgO.2SiO2', 'CMS2', 988, 1148, 1187, 1664, -2, 3, 0.000221,
        1613.93375497914, -4.79, 4874.5, 689.9
    ],
]

table = pd.DataFrame(table_values, columns=table_columns)
index_set = set(table.index)
len_table = len(table)

### Creating results dictionary

results = {}

for GS_name in GS_names:

    results[GS_name] = {
        'r': [],
        'abs_residual': [],
        'missing_Rc': [],
        'predicted_Rc': [],
    }

### Bootstrap calculations

for i in range(NUM_BOOTSTRAPS):

    # Generating the bootstrap
    sample = table.sample(len_table, replace=True)
    missing_indexes = index_set - set(sample.index)
    missing = table.loc[missing_indexes]

    # Adding noise
    boot_dic = {}
    missing_dic = {}

    while True:
        for param in std_abs:
            noise = normal(0, std_abs[param], len_table)
            boot_dic[param] = sample[param].values + noise

            noise = normal(0, std_abs[param], len(missing))
            missing_dic[param] = missing[param].values + noise

        logic1 = boot_dic['TUmax'] >= boot_dic['Tl']
        logic2 = missing_dic['TUmax'] >= missing_dic['Tl']

        if not all(logic1) and not all(logic2):
            break

    for param in std_pct:
        noise = normal(1, std_pct[param] / 100, len_table)
        boot_dic[param] = sample[param].values * noise

        noise = normal(1, std_pct[param] / 100, len(missing))
        missing_dic[param] = missing[param].values * noise

    noise = normal(0, log_visc_abs, len_table)
    boot_dic['log_viscosity_Tl'] = sample['ninf'].values + sample['A'].values \
        / (boot_dic['Tl'] - sample['T0'].values) + noise

    noise = normal(0, log_visc_abs, len(missing))
    missing_dic['log_viscosity_Tl'] = missing['ninf'].values \
        + missing['A'].values \
        / (missing_dic['Tl'] - missing['T0'].values) + noise

    # Compute Rc
    Rc = logRcCalc(**boot_dic)
    missing_Rc = logRcCalc(**missing_dic)

    # For each GS...
    for GS_name in GS_names:

        GS = function_dic[GS_name](**boot_dic)
        slope, intercept, r_value, _, _ = linregress(x=GS, y=Rc)

        # Prediction
        missing_GS = function_dic[GS_name](**missing_dic)
        predicted_Rc = slope * missing_GS + intercept

        # Errors
        results[GS_name]['r'].append(r_value)

        residual = missing_Rc - predicted_Rc
        results[GS_name]['missing_Rc'].extend(missing_Rc)
        results[GS_name]['predicted_Rc'].extend(predicted_Rc)

        abs_residual = abs(residual)
        results[GS_name]['abs_residual'].extend(abs_residual)

### Wilcoxon test

wilcoxon_table = np.zeros((len(GS_names), len(GS_names)))
measure = 'abs_residual'

for index1, index2 in it.combinations(np.arange(len(GS_names)), 2):

    GS1 = GS_names[index1]
    GS2 = GS_names[index2]

    residual1 = results[GS1][measure]
    residual2 = results[GS2][measure]

    w, p = wilcoxon(residual1, residual2)

    if p < (100 - CONFIDENCE) / 100:

        w, p = wilcoxon(residual1, residual2, alternative='greater')

        if p < (100 - CONFIDENCE) / 100:
            wilcoxon_table[index1, index2] = -1
            wilcoxon_table[index2, index1] = 1

        else:
            wilcoxon_table[index1, index2] = 1
            wilcoxon_table[index2, index1] = -1

R2mode = []
for GS_name in GS_names:
    R2 = np.array(results[GS_name]['r'])**2
    R2mode.append(mode(np.around(R2, 2))[0][0])

### Tabela Wilcoxon

wilcoxon_table = pd.DataFrame(wilcoxon_table, columns=GS_names, index=GS_names)
wilcoxon_table['sum'] = wilcoxon_table.sum(axis=1)
wilcoxon_table['R2_mode'] = R2mode

wilcoxon_table.sort_values(by='sum', ascending=False, inplace=True)
wilcoxon_table = wilcoxon_table[[
    *wilcoxon_table.index,
    'R2_mode',
]]

# wilcoxon_table is the final result. It can be exported with:
# wilcoxon_table.to_excel('export_file_path.xlsx')
