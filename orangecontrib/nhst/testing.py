"""Test helpers shared by the stats and widget tests (no Qt imports)."""
from Orange.data import Table


def reference_iris():
    """Orange's iris with the two UCI-erratum rows (0-based 34 and 37) patched.

    The spec's reference values use Fisher's corrected data (= sklearn's load_iris).
    """
    t = Table("iris")
    X = t.X.copy()
    X[34] = [4.9, 3.1, 1.5, 0.2]
    X[37] = [4.9, 3.6, 1.4, 0.1]
    return Table.from_numpy(t.domain, X, t.Y, t.metas)
