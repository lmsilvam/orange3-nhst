from dataclasses import dataclass

import numpy as np
from scipy import stats

from .descriptives import check_sample, describe
from . import effect_sizes as es


@dataclass
class OneSampleResult:
    n: int
    mean: float
    sd: float
    se: float
    mu0: float
    t: float
    df: float
    p: float
    mean_diff: float
    ci_low: float
    ci_high: float
    d: float


@dataclass
class IndependentResult:
    test: str  # "Welch" or "Student"
    n1: int
    n2: int
    mean1: float
    mean2: float
    sd1: float
    sd2: float
    t: float
    df: float
    p: float
    mean_diff: float  # group 1 minus group 2
    ci_low: float
    ci_high: float
    d: float  # Cohen's d (pooled SD)
    g: float  # Hedges' g


@dataclass
class PairedResult:
    n: int
    mean1: float
    mean2: float
    mean_diff: float  # variable 1 minus variable 2
    sd_diff: float
    se_diff: float
    t: float
    df: float
    p: float
    ci_low: float
    ci_high: float
    dz: float


def one_sample_t(x, mu0=0.0, alternative="two-sided", confidence=0.95):
    x = check_sample(x, "Variable")
    d = describe(x)
    res = stats.ttest_1samp(x - mu0, 0.0, alternative=alternative)
    ci = res.confidence_interval(confidence)  # CI of the mean difference
    return OneSampleResult(d.n, d.mean, d.sd, d.se, mu0, float(res.statistic),
                           float(res.df), float(res.pvalue), d.mean - mu0,
                           float(ci.low), float(ci.high),
                           es.cohens_d_one_sample(x, mu0))


def independent_t(x1, x2, equal_var=False, alternative="two-sided", confidence=0.95):
    x1 = check_sample(x1, "Group 1")
    x2 = check_sample(x2, "Group 2")
    d1, d2 = describe(x1), describe(x2)
    res = stats.ttest_ind(x1, x2, equal_var=equal_var, alternative=alternative)
    ci = res.confidence_interval(confidence)
    return IndependentResult(
        "Student" if equal_var else "Welch", d1.n, d2.n, d1.mean, d2.mean,
        d1.sd, d2.sd, float(res.statistic), float(res.df), float(res.pvalue),
        d1.mean - d2.mean, float(ci.low), float(ci.high),
        es.cohens_d(x1, x2), es.hedges_g(x1, x2))


def paired_t(x1, x2, alternative="two-sided", confidence=0.95):
    x1 = np.asarray(x1, dtype=float).ravel()
    x2 = np.asarray(x2, dtype=float).ravel()
    if x1.shape != x2.shape:
        raise ValueError("Variables must have the same length.")
    ok = ~(np.isnan(x1) | np.isnan(x2))  # listwise per pair
    x1, x2 = x1[ok], x2[ok]
    diff = check_sample(x1 - x2, "Difference")
    d = describe(diff)
    res = stats.ttest_rel(x1, x2, alternative=alternative)
    ci = res.confidence_interval(confidence)
    return PairedResult(d.n, float(np.mean(x1)), float(np.mean(x2)), d.mean,
                        d.sd, d.se, float(res.statistic), float(res.df),
                        float(res.pvalue), float(ci.low), float(ci.high),
                        es.cohens_dz(diff))
