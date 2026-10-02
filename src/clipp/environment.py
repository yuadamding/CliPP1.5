"""Small, secret-free numerical environment receipt; no dependency installation."""

from contextlib import redirect_stdout
from importlib.metadata import version
import io
import os
import platform
import sys
import warnings

import numpy as np
import scipy


def fingerprint():
    configuration = {}
    configuration_warnings = {}
    for name, module in (("numpy", np), ("scipy", scipy)):
        stream = io.StringIO()
        with redirect_stdout(stream), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            module.show_config()
        configuration[name] = stream.getvalue()
        configuration_warnings[name] = [str(item.message) for item in caught]
    try:
        from threadpoolctl import threadpool_info

        libraries = threadpool_info()
    except ImportError:
        libraries = None
    return {
        "python": platform.python_version(),
        "python_build": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "dependencies": {name: version(name) for name in ("numpy", "scipy", "pandas")},
        "threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_CORETYPE")
        },
        "numerical_builds": configuration,
        "numerical_build_warnings": configuration_warnings,
        "loaded_blas": libraries,
        "loaded_blas_status": "recorded" if libraries is not None else "threadpoolctl not installed",
    }
