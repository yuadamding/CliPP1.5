"""CUDA public entry point and recursive source provenance.

Historical chain implementations are not a production fallback. Use the pinned
pre-migration revision to reproduce their API or chain-specific benchmarks.
"""
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import scipy
import torch
from . import __version__


def _floating_format(dtype):
    """Source-compatible precision metadata; no numerical fitting occurs here."""
    info = np.finfo(dtype)
    return {"dtype": np.dtype(dtype).name, "storage_bits": int(np.dtype(dtype).itemsize * 8),
            "nmant": int(info.nmant), "significand_bits": int(info.nmant + 1),
            "exponent_bits": int(info.iexp), "minexp": int(info.minexp), "maxexp": int(info.maxexp),
            "machep": int(info.machep), "eps": str(info.eps), "epsneg": str(info.epsneg),
            "smallest_normal": str(info.smallest_normal), "max_finite": str(info.max)}


def source_provenance():
    package = Path(__file__).resolve().parent
    hashes = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(package.rglob("*.py"))}
    digest = hashlib.sha256()
    for name, value in hashes.items():
        digest.update(f"{name}\0{value}\n".encode())
    commit = None
    root = package.parent.parent
    if (root / ".git").exists():
        proc = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        if proc.returncode == 0:
            commit = proc.stdout.strip()
    return dict(package_version=__version__, source_commit=commit, source_sha256=digest.hexdigest(),
                source_files=hashes, python=platform.python_version(), interpreter=sys.executable,
                numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__, dtype="float64",
                precision_scope="PyTorch CUDA inference uses float64. This receipt does not qualify other platforms.",
                platform=platform.platform(), machine=platform.machine(),
                floating_point_formats={"float64": _floating_format(np.float64),
                                        "longdouble": _floating_format(np.longdouble)})


def fit(input_file, outdir=None, *, max_major_cn=4, verbose=False, device="cuda:0"):
    from .cuda_api import fit as implementation
    return implementation(input_file, outdir, max_major_cn=max_major_cn, verbose=verbose, device=device)
