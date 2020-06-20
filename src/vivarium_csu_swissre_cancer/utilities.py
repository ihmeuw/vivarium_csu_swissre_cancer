import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from vivarium_public_health.risks.data_transformations import pivot_categorical

from vivarium_csu_swissre_cancer import globals as project_globals


class TruncnormParams:
    def __init__(self, mean, sd, lower=0, upper=1):
        self.a = (lower - mean) / sd if sd else mean
        self.b = (upper - mean) / sd if sd else mean
        self.loc = mean
        self.scale = sd


def sample_truncnorm_distribution(seed: int, mean: float, sd: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Gets a single random draw from a truncated normal distribution.
    Parameters
    ----------
    seed
        Seed for the random number generator.
    mean
        mean of truncnorm distribution
    sd
        standard deviation of truncnorm distribution
    lower
        lower bound of truncnorm distribution
    upper
        upper bound of truncnorm distribution
    Returns
    -------
        The random variate from the truncated normal distribution.
    """
    # Handle degenerate distribution
    if not sd:
        return mean

    np.random.seed(seed)
    params = TruncnormParams(mean, sd, lower, upper)
    return truncnorm.rvs(params.a, params.b, params.loc, params.scale)


def sanitize_location(location: str):
    """Cleans up location formatting for writing and reading from file names.

    Parameters
    ----------
    location
        The unsanitized location name.

    Returns
    -------
        The sanitized location name (lower-case with white-space and
        special characters removed.

    """
    # FIXME: Should make this a reversible transformation.
    return location.replace(" ", "_").replace("'", "_").lower()
