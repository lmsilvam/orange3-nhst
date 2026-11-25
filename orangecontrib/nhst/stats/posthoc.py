from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats

from .anova import prepare_groups


@dataclass
class PostHocRow:
    group_a: str
    group_b: str
    mean_diff: float  # A minus B
    ci_low: float
    ci_high: float
    p_adj: float


def _setup(groups, labels):
    labels = list(labels) if labels is not None else [f"Group {i + 1}" for i in range(len(groups))]
    return prepare_groups(groups, labels), labels


def tukey_hsd(groups, labels=None, confidence=0.95):
    gs, labels = _setup(groups, labels)
    res = stats.tukey_hsd(*gs)
    ci = res.confidence_interval(confidence)
    return [PostHocRow(labels[i], labels[j], float(res.statistic[i, j]),
                       float(ci.low[i, j]), float(ci.high[i, j]),
                       float(res.pvalue[i, j]))
            for i, j in combinations(range(len(gs)), 2)]


def games_howell(groups, labels=None, confidence=0.95):
    gs, labels = _setup(groups, labels)
    k = len(gs)
    q = stats.studentized_range
    rows = []
    for i, j in combinations(range(k), 2):
        a, b = gs[i], gs[j]
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        se = np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        diff = a.mean() - b.mean()
        p = q.sf(abs(diff) / se * np.sqrt(2), k, df)
        half = q.ppf(confidence, k, df) / np.sqrt(2) * se
        rows.append(PostHocRow(labels[i], labels[j], float(diff), float(diff - half),
                               float(diff + half), float(p)))
    return rows


def bonferroni(groups, labels=None, welch=False, confidence=0.95):
    """Pairwise t-tests (pooled variance over all groups, or Welch), p*m capped at 1.

    CI uses the Bonferroni-adjusted level 1 - (1 - confidence) / m.
    """
    gs, labels = _setup(groups, labels)
    k = len(gs)
    pairs = list(combinations(range(k), 2))
    m = len(pairs)
    level = 1 - (1 - confidence) / m
    N = sum(len(g) for g in gs)
    ms_w = sum((len(g) - 1) * g.var(ddof=1) for g in gs) / (N - k)
    rows = []
    for i, j in pairs:
        a, b = gs[i], gs[j]
        diff = a.mean() - b.mean()
        if welch:
            res = stats.ttest_ind(a, b, equal_var=False)
            ci = res.confidence_interval(level)
            low, high, p = ci.low, ci.high, res.pvalue
        else:
            se = np.sqrt(ms_w * (1 / len(a) + 1 / len(b)))
            df = N - k
            h = stats.t.ppf(0.5 + level / 2, df) * se
            low, high, p = diff - h, diff + h, 2 * stats.t.sf(abs(diff) / se, df)
        rows.append(PostHocRow(labels[i], labels[j], float(diff), float(low),
                               float(high), min(1.0, float(p) * m)))
    return rows
