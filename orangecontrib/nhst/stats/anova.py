from dataclasses import dataclass

import numpy as np
from scipy import stats

from .descriptives import check_sample
from . import effect_sizes as es


@dataclass
class AnovaResult:
    method: str  # "Classic" or "Welch"
    k: int
    f: float
    df1: float
    df2: float
    p: float
    # Classic sums of squares; eta2/omega2 always come from these, also for Welch.
    ss_between: float
    ss_within: float
    ms_between: float
    ms_within: float
    eta2: float
    omega2: float


def prepare_groups(groups, labels=None):
    """Drop NaN per group; raise ValueError for <2 groups, n<2 or zero variance."""
    labels = labels or [f"Group {i + 1}" for i in range(len(groups))]
    if len(groups) < 2:
        raise ValueError("At least 2 groups are required.")
    return [check_sample(g, f"Group '{lab}'") for g, lab in zip(groups, labels)]


def one_way_anova(groups, method="classic", labels=None):
    gs = prepare_groups(groups, labels)
    k = len(gs)
    n = np.array([len(g) for g in gs], dtype=float)
    m = np.array([g.mean() for g in gs])
    v = np.array([g.var(ddof=1) for g in gs])
    N = n.sum()
    grand = np.concatenate(gs).mean()
    ss_b = float(np.sum(n * (m - grand) ** 2))
    ss_w = float(np.sum((n - 1) * v))
    ms_b, ms_w = ss_b / (k - 1), ss_w / (N - k)
    if method == "welch":
        w = n / v
        gm = np.sum(w * m) / w.sum()
        a = np.sum(w * (m - gm) ** 2) / (k - 1)
        lam = np.sum((1 - w / w.sum()) ** 2 / (n - 1))
        b = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam
        f, df1, df2 = a / b, k - 1, (k ** 2 - 1) / (3 * lam)
        p = stats.f.sf(f, df1, df2)
        name = "Welch"
    else:
        f, p = stats.f_oneway(*gs)
        df1, df2, name = k - 1, N - k, "Classic"
    return AnovaResult(name, k, float(f), float(df1), float(df2), float(p),
                       ss_b, ss_w, ms_b, ms_w,
                       es.eta_squared(ss_b, ss_b + ss_w),
                       es.omega_squared(ss_b, ss_b + ss_w, k - 1, ms_w))
