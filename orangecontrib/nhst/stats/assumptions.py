from dataclasses import dataclass

import numpy as np
from scipy import stats

from .anova import prepare_groups
from .descriptives import clean


@dataclass
class NormalityResult:
    n: int
    w: float
    p: float
    note: str = ""  # non-empty when the test was skipped


@dataclass
class LeveneResult:
    f: float
    p: float


def shapiro_wilk(x):
    x = clean(x)
    n = len(x)
    if n < 3 or n > 5000:
        return NormalityResult(n, np.nan, np.nan,
                               "Shapiro-Wilk skipped: N must be between 3 and 5000.")
    if np.ptp(x) == 0:
        return NormalityResult(n, np.nan, np.nan, "Shapiro-Wilk skipped: zero variance.")
    res = stats.shapiro(x)
    return NormalityResult(n, float(res.statistic), float(res.pvalue))


def levene_median(groups, labels=None):
    gs = prepare_groups(groups, labels)
    res = stats.levene(*gs, center="median")
    return LeveneResult(float(res.statistic), float(res.pvalue))


def shapiro_residuals(groups, labels=None):
    """Shapiro-Wilk on ANOVA residuals (observation minus its group mean)."""
    gs = prepare_groups(groups, labels)
    return shapiro_wilk(np.concatenate([g - g.mean() for g in gs]))
