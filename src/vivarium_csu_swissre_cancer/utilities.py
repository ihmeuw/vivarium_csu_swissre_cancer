import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from vivarium.framework.randomness import get_hash


class TruncnormDist:
    """Defines an instance of a truncated normal distribution.
    Parameters
    ----------
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
        An object with parameters for scipy.stats.truncnorm
    """
    def __init__(self, name, mean, sd, lower=0, upper=1, key=None):
        self.name = name
        self.a = (lower - mean) / sd if sd else 0
        self.b = (upper - mean) / sd if sd else 0
        self.mean = mean
        self.sd = sd
        self.key = key if key else name
        
    def get_random_variable(self, draw: int) -> float:
        """Gets a single random draw from a truncated normal distribution.
        Parameters
        ----------
        draw
            Draw for this simulation
        Returns
        -------
            The random variate from the truncated normal distribution.
        """
        # Handle degenerate distribution
        if not self.sd:
            return self.mean
    
        np.random.seed(get_hash(f'{self.key}_draw_{draw}'))
        return truncnorm.rvs(self.a, self.b, self.mean, self.sd)

    def ppf(self, quantiles: pd.Series) -> pd.Series:
        return truncnorm(self.a, self.b, self.mean, self.sd).ppf(quantiles)


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
