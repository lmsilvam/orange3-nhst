import numpy as np
import pytest
from scipy import stats

from orangecontrib.nhst.stats.ttests import independent_t, one_sample_t, paired_t

RNG = np.random.default_rng(0)


def test_one_sample_reference(iris):
    # spec reference values (iris, df = 149)
    r = one_sample_t(iris["sepal length"], 5.5)
    assert r.n == 150 and r.df == 149
    assert r.t == pytest.approx(5.078, abs=1e-3)
    assert r.p == pytest.approx(1.12e-06, rel=0.01)
    assert r.mean_diff == pytest.approx(0.343, abs=1e-3)
    assert (r.ci_low, r.ci_high) == (pytest.approx(0.210, abs=1e-3), pytest.approx(0.477, abs=1e-3))
    assert r.d == pytest.approx(0.415, abs=1e-3)
    assert r.sd == pytest.approx(0.828, abs=1e-3) and r.se == pytest.approx(0.068, abs=1e-3)

    r = one_sample_t(iris["sepal width"], 3.0)
    assert (r.t, r.p, r.mean_diff, r.d) == pytest.approx((1.611, 0.109, 0.057, 0.132), abs=1e-3)
    assert (r.ci_low, r.ci_high) == pytest.approx((-0.013, 0.128), abs=1e-3)

    r = one_sample_t(iris["petal length"], 3.5)
    assert (r.t, r.p, r.mean_diff, r.d) == pytest.approx((1.790, 0.0755, 0.258, 0.146), abs=1e-3)
    assert (r.ci_low, r.ci_high) == pytest.approx((-0.027, 0.543), abs=1e-3)


def test_one_sample_manual_crosscheck():
    # cross-check: textbook formula t = (m - mu0) / (s / sqrt(n)), CI with t quantile
    x = RNG.normal(5, 2, 30)
    r = one_sample_t(x, 4.0, confidence=0.9)
    se = x.std(ddof=1) / np.sqrt(30)
    assert r.t == pytest.approx((x.mean() - 4) / se)
    h = stats.t.ppf(0.95, 29) * se
    assert (r.ci_low, r.ci_high) == pytest.approx((x.mean() - 4 - h, x.mean() - 4 + h))


@pytest.mark.parametrize("mu0, agree, other", [(0.0, "greater", "less"), (3.0, "less", "greater")])
def test_one_sample_one_sided_half_of_two_sided(mu0, agree, other):
    # cross-check: one-sided p = two-sided p / 2 when sign of t agrees with alternative
    x = RNG.normal(1, 1, 25)
    two = one_sample_t(x, mu0)
    assert one_sample_t(x, mu0, alternative=agree).p == pytest.approx(two.p / 2)
    assert one_sample_t(x, mu0, alternative=other).p == pytest.approx(1 - two.p / 2)


def test_one_sample_one_sided_ci_is_half_open():
    x = RNG.normal(1, 1, 25)
    assert one_sample_t(x, 0.0, alternative="greater").ci_high == np.inf
    assert one_sample_t(x, 0.0, alternative="less").ci_low == -np.inf


def test_one_sample_errors():
    with pytest.raises(ValueError, match="fewer than 2"):
        one_sample_t([1.0, np.nan, np.nan])
    with pytest.raises(ValueError, match="zero variance"):
        one_sample_t([2.0, 2.0, 2.0])


def test_one_sample_n2_and_nan():
    r = one_sample_t([1.0, 3.0, np.nan], 0.0)  # N=2 after NaN drop
    assert r.n == 2 and r.df == 1 and r.mean == 2.0


def test_independent_reference(iris):
    sp = iris["species"]
    a = iris["sepal width"][sp == "Iris-setosa"]
    b = iris["sepal width"][sp == "Iris-versicolor"]
    w = independent_t(a, b)
    assert w.test == "Welch"
    assert (w.n1, w.n2) == (50, 50)
    assert (w.mean1, w.mean2) == pytest.approx((3.428, 2.770), abs=1e-3)
    assert (w.sd1, w.sd2) == pytest.approx((0.379, 0.314), abs=1e-3)
    assert w.t == pytest.approx(9.455, abs=1e-3)
    assert w.df == pytest.approx(94.70, abs=0.01)
    assert w.p == pytest.approx(2.48e-15, rel=0.01)
    assert w.mean_diff == pytest.approx(0.658, abs=1e-3)
    assert (w.ci_low, w.ci_high) == pytest.approx((0.520, 0.796), abs=1e-3)
    s = independent_t(a, b, equal_var=True)
    assert s.test == "Student"
    assert s.t == pytest.approx(9.455, abs=1e-3) and s.df == 98
    assert s.p == pytest.approx(1.85e-15, rel=0.01)
    assert s.d == pytest.approx(1.891, abs=1e-3)
    assert s.g == pytest.approx(1.876, abs=1e-3)


def test_independent_student_ci_and_g_crosscheck():
    # cross-check: pooled-variance CI by hand; g = d * J with J = 1 - 3/(4 df - 1)
    a, b = RNG.normal(0, 1, 12), RNG.normal(1, 2, 20)
    r = independent_t(a, b, equal_var=True, confidence=0.9)
    sp2 = (11 * a.var(ddof=1) + 19 * b.var(ddof=1)) / 30
    se = np.sqrt(sp2 * (1 / 12 + 1 / 20))
    h = stats.t.ppf(0.95, 30) * se
    assert (r.ci_low, r.ci_high) == pytest.approx((a.mean() - b.mean() - h, a.mean() - b.mean() + h))
    assert r.d == pytest.approx((a.mean() - b.mean()) / np.sqrt(sp2))
    assert r.g == pytest.approx(r.d * (1 - 3 / (4 * 30 - 1)))


def test_independent_unequal_sizes_welch_df():
    # cross-check: Welch-Satterthwaite df by hand
    a, b = RNG.normal(0, 1, 8), RNG.normal(0, 3, 31)
    r = independent_t(a, b)
    va, vb = a.var(ddof=1) / 8, b.var(ddof=1) / 31
    assert r.df == pytest.approx((va + vb) ** 2 / (va ** 2 / 7 + vb ** 2 / 30))


@pytest.mark.parametrize("equal_var", [True, False])
def test_independent_one_sided_half_two_sided(equal_var):
    # cross-check: one-sided p = two-sided p / 2 when sign agrees
    a, b = RNG.normal(1, 1, 15), RNG.normal(0, 1, 18)
    two = independent_t(a, b, equal_var=equal_var)
    assert two.t > 0
    assert independent_t(a, b, equal_var, "greater").p == pytest.approx(two.p / 2)
    assert independent_t(a, b, equal_var, "less").p == pytest.approx(1 - two.p / 2)


def test_independent_errors_and_nan():
    with pytest.raises(ValueError, match="Group 1"):
        independent_t([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="Group 2.*zero variance"):
        independent_t([1.0, 2.0], [5.0, 5.0])
    r = independent_t([1.0, 2.0, np.nan, 4.0], [2.0, 5.0, 9.0])
    assert (r.n1, r.n2) == (3, 3)


def test_paired_reference(iris):
    r = paired_t(iris["sepal length"], iris["petal length"])
    assert r.n == 150 and r.df == 149
    assert r.mean_diff == pytest.approx(2.085, abs=1e-3)
    assert r.sd_diff == pytest.approx(1.120, abs=1e-3)
    assert r.se_diff == pytest.approx(0.091, abs=1e-3)
    assert r.t == pytest.approx(22.813, abs=1e-3)
    assert r.p == pytest.approx(1.80e-50, rel=0.01)
    assert (r.ci_low, r.ci_high) == pytest.approx((1.905, 2.266), abs=1e-3)
    assert r.dz == pytest.approx(1.863, abs=1e-3)


def test_paired_equals_one_sample_on_differences():
    # cross-check: paired t == one-sample t on differences
    a, b = RNG.normal(0, 1, 20), RNG.normal(0.5, 1, 20)
    p, o = paired_t(a, b), one_sample_t(a - b, 0.0)
    assert (p.t, p.p, p.ci_low, p.ci_high, p.dz) == pytest.approx((o.t, o.p, o.ci_low, o.ci_high, o.d))


def test_paired_listwise_nan_and_errors():
    r = paired_t([1.0, 2.0, np.nan, 5.0, 7.0], [2.0, np.nan, 3.0, 4.0, 4.0])
    assert r.n == 3 and r.mean1 == pytest.approx((1 + 5 + 7) / 3) and r.mean2 == pytest.approx(10 / 3)
    with pytest.raises(ValueError, match="fewer than 2"):
        paired_t([1.0, 2.0, np.nan], [1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="zero variance"):
        paired_t([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])  # constant difference
    with pytest.raises(ValueError, match="same length"):
        paired_t([1.0, 2.0], [1.0])


def test_paired_one_sided():
    a, b = RNG.normal(1, 1, 20), RNG.normal(0, 1, 20)
    two = paired_t(a, b)
    assert two.t > 0
    assert paired_t(a, b, "greater").p == pytest.approx(two.p / 2)
