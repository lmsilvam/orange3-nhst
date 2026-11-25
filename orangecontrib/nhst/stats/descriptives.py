from dataclasses import dataclass

import numpy as np


@dataclass
class Descriptives:
    n: int
    mean: float
    sd: float
    se: float


def clean(x):
    """Return x as a float array with NaN dropped."""
    x = np.asarray(x, dtype=float).ravel()
    return x[~np.isnan(x)]


def describe(x):
    x = clean(x)
    n = len(x)
    if n == 0:
        return Descriptives(0, np.nan, np.nan, np.nan)
    sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
    return Descriptives(n, float(np.mean(x)), sd, sd / np.sqrt(n))


def check_sample(x, label="Sample"):
    """Clean x and raise ValueError if a t/ANOVA statistic would be undefined."""
    x = clean(x)
    if len(x) < 2:
        raise ValueError(f"{label} has fewer than 2 valid observations.")
    if np.ptp(x) == 0:
        raise ValueError(f"{label} has zero variance.")
    return x
