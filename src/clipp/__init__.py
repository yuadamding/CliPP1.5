"""CliPP1.5 single-sample fixed-chain estimator."""

__version__ = "1.6.0"


def fit(*args, **kwargs):
    """Fit inputs to a new output directory; see :func:`clipp.api.fit`."""
    from .api import fit as run

    return run(*args, **kwargs)


def load_result(directory):
    """Independently verify and read a compact saved result."""
    from .output import load_result as read

    return read(directory)


def validate_inputs(snv_input, cn_input, purity_input):
    """Check the production input contract without fitting."""
    from .preprocessing import validate_inputs as validate

    return validate(snv_input, cn_input, purity_input)
