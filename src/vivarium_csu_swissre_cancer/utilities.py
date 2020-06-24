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
    def __init__(self, name, mean, sd, lower=0, upper=1):
        self.name = name
        self.a = (lower - mean) / sd if sd else 0
        self.b = (upper - mean) / sd if sd else 0
        self.loc = mean
        self.scale = sd
        
    def sample_screening_parameter(self, draw: int) -> float:
        """Gets a single random draw from a truncated normal distribution.
        Parameters
        ----------
        draw
            Draw for this simulation
        params
            TruncnorParams object with parameters for truncated normal distribution
        Returns
        -------
            The random variate from the truncated normal distribution.
        """
        # Handle degenerate distribution
        if not self.scale:
            return self.loc
    
        np.random.seed(get_hash(f'{self.name}_draw_{draw}'))
        return truncnorm.rvs(self.a, self.b, self.loc, self.scale)

    def get_draw(self, quantiles: pd.Series) -> pd.Series:
        return truncnorm(self.a, self.b, self.loc, self.scale).ppf(quantiles)


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
