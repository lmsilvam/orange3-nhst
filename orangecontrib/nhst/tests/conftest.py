import numpy as np
import pytest

from orangecontrib.nhst.testing import reference_iris


@pytest.fixture(scope="session")
def iris():
    """dict: column name -> numpy array, plus 'species' string array."""
    t = reference_iris()
    X = t.X
    out = {v.name: X[:, i] for i, v in enumerate(t.domain.attributes)}
    out["species"] = np.array([t.domain.class_var.values[int(i)] for i in t.Y])
    return out


@pytest.fixture(scope="session")
def sepal_length_groups(iris):
    sp = iris["species"]
    names = list(dict.fromkeys(sp))
    return names, [iris["sepal length"][sp == n] for n in names]
