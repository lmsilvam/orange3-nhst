import numpy as np
import pytest

from orangecontrib.nhst.stats.descriptives import check_sample, describe


def test_describe_numpy_crosscheck():
    # cross-check: plain numpy
    x = np.array([1.0, 2.0, 4.0, 8.0])
    d = describe(x)
    assert d.n == 4
    assert d.mean == pytest.approx(3.75)
    assert d.sd == pytest.approx(np.std(x, ddof=1))
    assert d.se == pytest.approx(np.std(x, ddof=1) / 2)


def test_describe_drops_nan():
    d = describe([1, np.nan, 3, 5])
    assert d.n == 3 and d.mean == pytest.approx(3.0)


def test_describe_single_obs_sd_nan():
    d = describe([2.0])
    assert d.n == 1 and np.isnan(d.sd) and np.isnan(d.se)


def test_describe_iris(iris):
    # reference values from spec (sepal length, N=150)
    d = describe(iris["sepal length"])
    assert (d.n, round(d.mean, 3), round(d.sd, 3), round(d.se, 3)) == (150, 5.843, 0.828, 0.068)


def test_check_sample_errors():
    with pytest.raises(ValueError, match="fewer than 2"):
        check_sample([1.0, np.nan], "G")
    with pytest.raises(ValueError, match="zero variance"):
        check_sample([3.0, 3.0, 3.0], "G")
