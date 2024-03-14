"""onion-clustering."""

from onion_clustering.main import main as main
from onion_clustering.main_2d import main as main_2d


class Onion1D:
    """Performs onion clustering on univariate time-series."""

    def __init__(self, full_output=True):
        self.output = main(full_output)


class Onion2D:
    """Performs onion clustering on multivariate time-series."""

    def __init__(self, full_output=True):
        self.output = main_2d(full_output)
