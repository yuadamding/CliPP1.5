"""Single-sample complete-graph CUDA inference."""
__version__ = "0.5.0.dev0"


def fit(input_file, outdir=None, *, max_major_cn=4, verbose=False, device="cuda:0"):
    from .api import fit as implementation
    return implementation(input_file, outdir, max_major_cn=max_major_cn, verbose=verbose, device=device)


def __getattr__(name):
    if name == "QualificationError":
        from .cuda.policy import QualificationError
        return QualificationError
    if name == "FitResult":
        from .cuda_api import FitResult
        return FitResult
    if name in {"InputError", "FitError", "NoEligibleMutationsError", "NumericalQualificationError"}:
        from . import types
        return getattr(types, name)
    raise AttributeError(name)
