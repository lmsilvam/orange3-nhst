from itertools import combinations

import numpy as np
import pytest
from scipy import stats

from orangecontrib.nhst.stats.anova import one_way_anova
from orangecontrib.nhst.stats.assumptions import levene_median, shapiro_residuals, shapiro_wilk
from orangecontrib.nhst.stats.posthoc import bonferroni, games_howell, tukey_hsd
from orangecontrib.nhst.stats.ttests import independent_t

RNG = np.random.default_rng(1)


def test_classic_reference(sepal_length_groups):
    # spec reference values
    _, gs = sepal_length_groups
    r = one_way_anova(gs)
    assert r.method == "Classic" and (r.df1, r.df2) == (2, 147)
    assert r.f == pytest.approx(119.265, abs=1e-3)
    assert r.p == pytest.approx(1.67e-31, rel=0.01)
    assert r.ss_between == pytest.approx(63.212, abs=1e-3)
    assert r.ss_within == pytest.approx(38.956, abs=1e-3)
    assert r.ms_between == pytest.approx(r.ss_between / 2)
    assert r.ms_within == pytest.approx(r.ss_within / 147)
    assert r.eta2 == pytest.approx(0.619, abs=1e-3)
    assert r.omega2 == pytest.approx(0.612, abs=1e-3)


def test_welch_reference(sepal_length_groups):
    _, gs = sepal_length_groups
    r = one_way_anova(gs, "welch")
    assert r.method == "Welch" and r.df1 == 2
    assert r.f == pytest.approx(138.908, abs=1e-3)
    assert r.df2 == pytest.approx(92.21, abs=0.01)
    assert r.p == pytest.approx(1.51e-28, rel=0.01)


def test_two_groups_f_equals_t_squared():
    # cross-check: classic F == Student t^2 (and Welch ANOVA F == Welch t^2)
    a, b = RNG.normal(0, 1, 14), RNG.normal(1, 2, 23)
    t = independent_t(a, b, equal_var=True)
    f = one_way_anova([a, b])
    assert f.f == pytest.approx(t.t ** 2) and f.p == pytest.approx(t.p)
    assert f.df2 == t.df
    tw = independent_t(a, b)
    fw = one_way_anova([a, b], "welch")
    assert fw.f == pytest.approx(tw.t ** 2) and fw.df2 == pytest.approx(tw.df)
    assert fw.p == pytest.approx(tw.p)


def test_anova_ss_decomposition_unequal_sizes():
    # cross-check: SS_total computed directly from pooled data
    gs = [RNG.normal(i, 1, n) for i, n in enumerate([5, 9, 17, 4])]
    r = one_way_anova(gs)
    allx = np.concatenate(gs)
    assert r.ss_between + r.ss_within == pytest.approx(np.sum((allx - allx.mean()) ** 2))
    assert r.df2 == len(allx) - 4


def test_anova_nan_and_errors():
    r = one_way_anova([[1.0, 2.0, np.nan, 3.0], [4.0, 5.0, 7.0]])
    assert r.df2 == 4
    with pytest.raises(ValueError, match="At least 2 groups"):
        one_way_anova([[1.0, 2.0]])
    with pytest.raises(ValueError, match="'B'.*fewer than 2"):
        one_way_anova([[1.0, 2.0], [3.0]], labels=["A", "B"])
    with pytest.raises(ValueError, match="'B'.*zero variance"):
        one_way_anova([[1.0, 2.0], [3.0, 3.0]], labels=["A", "B"])


def test_tukey_reference(sepal_length_groups):
    names, gs = sepal_length_groups
    rows = tukey_hsd(gs, names)
    exp = [(-0.930, -1.174, -0.686, 2.4e-14), (-1.582, -1.826, -1.338, 2.2e-14),
           (-0.652, -0.896, -0.408, 8.3e-09)]
    assert [(r.group_a, r.group_b) for r in rows] == [
        ("Iris-setosa", "Iris-versicolor"), ("Iris-setosa", "Iris-virginica"),
        ("Iris-versicolor", "Iris-virginica")]
    for r, (d, lo, hi, p) in zip(rows, exp):
        assert (r.mean_diff, r.ci_low, r.ci_high) == pytest.approx((d, lo, hi), abs=1e-3)
        assert r.p_adj == pytest.approx(p, rel=0.15)  # spec says "about"


def test_tukey_confidence_widens():
    gs = [RNG.normal(i, 1, 10) for i in range(3)]
    w90, w99 = tukey_hsd(gs, confidence=0.90)[0], tukey_hsd(gs, confidence=0.99)[0]
    assert w99.ci_high - w99.ci_low > w90.ci_high - w90.ci_low


def test_games_howell_vs_direct_formula():
    # cross-check: direct implementation, t and Welch df from scipy ttest_ind(equal_var=False),
    # p from studentized_range.sf(|t|*sqrt2, k, df), CI half-width = q_crit/sqrt2 * se
    gs = [RNG.normal(0, 1, 8), RNG.normal(1, 3, 15), RNG.normal(2, 0.5, 11)]
    k = len(gs)
    rows = games_howell(gs, confidence=0.9)
    for r, (i, j) in zip(rows, combinations(range(k), 2)):
        res = stats.ttest_ind(gs[i], gs[j], equal_var=False)
        diff = gs[i].mean() - gs[j].mean()
        se = diff / res.statistic
        p = stats.studentized_range.sf(abs(res.statistic) * np.sqrt(2), k, res.df)
        half = stats.studentized_range.ppf(0.9, k, res.df) / np.sqrt(2) * abs(se)
        assert r.p_adj == pytest.approx(p)
        assert r.mean_diff == pytest.approx(diff)
        assert (r.ci_low, r.ci_high) == pytest.approx((diff - half, diff + half))


def test_games_howell_equal_variance_close_to_tukey(sepal_length_groups):
    # sanity cross-check: same direction and same order of magnitude as Tukey on iris
    names, gs = sepal_length_groups
    for g, t in zip(games_howell(gs, names), tukey_hsd(gs, names)):
        assert g.mean_diff == pytest.approx(t.mean_diff)
        assert g.p_adj < 1e-6 and t.p_adj < 1e-6


def test_bonferroni_pooled_vs_manual_scipy_p(sepal_length_groups):
    # cross-check: pooled variance t-test p computed via scipy t distribution, times m
    names, gs = sepal_length_groups
    rows = bonferroni(gs, names)
    N, k = 150, 3
    ms_w = one_way_anova(gs).ms_within
    for r, (i, j) in zip(rows, combinations(range(3), 2)):
        diff = gs[i].mean() - gs[j].mean()
        t = diff / np.sqrt(ms_w * (1 / 50 + 1 / 50))
        assert r.p_adj == pytest.approx(min(1, 3 * 2 * stats.t.sf(abs(t), N - k)))
        assert r.mean_diff == pytest.approx(diff)
        assert r.ci_low < r.mean_diff < r.ci_high


def test_bonferroni_welch_vs_manual_scipy_p_and_cap():
    # cross-check: scipy ttest_ind(equal_var=False) p * m, capped at 1
    gs = [RNG.normal(0, 1, 10), RNG.normal(0.05, 3, 14), RNG.normal(0.1, 1, 9)]
    rows = bonferroni(gs, welch=True)
    for r, (i, j) in zip(rows, combinations(range(3), 2)):
        p = stats.ttest_ind(gs[i], gs[j], equal_var=False).pvalue
        assert r.p_adj == pytest.approx(min(1.0, 3 * p))
    big = bonferroni([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]])
    assert all(r.p_adj == 1.0 for r in big)


def test_bonferroni_two_groups_equals_student_p():
    # cross-check: m = 1, so p_adj == Student t-test p and CI == Student CI
    a, b = RNG.normal(0, 1, 12), RNG.normal(1, 1, 12)
    r = bonferroni([a, b])[0]
    t = independent_t(a, b, equal_var=True)
    assert r.p_adj == pytest.approx(t.p)
    assert (r.ci_low, r.ci_high) == pytest.approx((t.ci_low, t.ci_high))


# ---- assumptions ----

def test_levene_and_residual_shapiro_reference(sepal_length_groups):
    _, gs = sepal_length_groups
    lev = levene_median(gs)
    assert lev.f == pytest.approx(6.353, abs=1e-3) and lev.p == pytest.approx(0.0023, abs=1e-4)
    sh = shapiro_residuals(gs)
    assert sh.n == 150
    assert sh.w == pytest.approx(0.988, abs=1e-3) and sh.p == pytest.approx(0.219, abs=1e-3)


def test_levene_independent_reference_and_shapiro_groups(iris):
    sp = iris["species"]
    a = iris["sepal width"][sp == "Iris-setosa"]
    b = iris["sepal width"][sp == "Iris-versicolor"]
    lev = levene_median([a, b])
    assert lev.f == pytest.approx(0.591, abs=1e-3) and lev.p == pytest.approx(0.444, abs=1e-3)
    sa, sb = shapiro_wilk(a), shapiro_wilk(b)
    assert (sa.w, sa.p) == pytest.approx((0.972, 0.272), abs=1e-3)
    assert (sb.w, sb.p) == pytest.approx((0.974, 0.338), abs=1e-3)


def test_shapiro_reference_iris_petal_and_diff(iris):
    s = shapiro_wilk(iris["petal length"])
    assert s.w == pytest.approx(0.876, abs=1e-3) and s.p == pytest.approx(7.4e-10, rel=0.05)
    d = shapiro_wilk(iris["sepal length"] - iris["petal length"])
    assert d.w == pytest.approx(0.899, abs=1e-3) and d.p == pytest.approx(1.2e-08, rel=0.05)


def test_shapiro_skips():
    assert shapiro_wilk([1.0, 2.0]).note and np.isnan(shapiro_wilk([1.0, 2.0]).p)
    big = shapiro_wilk(RNG.normal(size=5001))
    assert big.note and np.isnan(big.w) and big.n == 5001
    assert not shapiro_wilk([1.0, 2.0, 4.0, np.nan]).note  # NaN dropped, N=3
    assert shapiro_wilk([2.0, 2.0, 2.0]).note
    assert shapiro_wilk(RNG.normal(size=5000)).note == ""
