import numpy as np


def pooled_sd(x1, x2):
    n1, n2 = len(x1), len(x2)
    v = ((n1 - 1) * np.var(x1, ddof=1) + (n2 - 1) * np.var(x2, ddof=1)) / (n1 + n2 - 2)
    return float(np.sqrt(v))


def cohens_d_one_sample(x, mu0):
    return float((np.mean(x) - mu0) / np.std(x, ddof=1))


def cohens_d(x1, x2):
    """Cohen's d for two independent samples (pooled SD), group 1 minus group 2."""
    return float((np.mean(x1) - np.mean(x2)) / pooled_sd(x1, x2))


def hedges_j(df):
    return 1 - 3 / (4 * df - 1)


def hedges_g(x1, x2):
    return cohens_d(x1, x2) * hedges_j(len(x1) + len(x2) - 2)


def cohens_dz(diff):
    return float(np.mean(diff) / np.std(diff, ddof=1))


def eta_squared(ss_between, ss_total):
    return float(ss_between / ss_total)


def omega_squared(ss_between, ss_total, df_between, ms_within):
    return float((ss_between - df_between * ms_within) / (ss_total + ms_within))
