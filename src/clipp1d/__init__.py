"""Independent single-sample adaptive-chain CCF inference."""

__version__ = "0.2.0"


def fit(input_file, outdir=None, *, max_major_cn=4, verbose=False):
    """Fit one tumor sample; optionally publish three TSVs and run.json."""
    from .api import fit as _fit

    return _fit(input_file, outdir, max_major_cn=max_major_cn, verbose=verbose)


__all__ = ["fit", "__version__"]
