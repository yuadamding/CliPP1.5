"""CliPP1.5 single-sample fixed-chain estimator."""

__version__ = "1.5.1"


def fit(*args, **kwargs):
    """Fit inputs to a new output directory; see :func:`clipp.api.fit`."""
    from .api import fit as run

    return run(*args, **kwargs)
