"""Explicit synthetic historical package staging for subprocess regression tests.

The legacy test API is installed only into a temporary test directory. Production
imports, public entry points, and current-source benchmark launches stay CUDA-only.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def stage_historical_package(source_root):
    source_root = Path(source_root)
    repo = Path(__file__).resolve().parents[1]
    package = source_root / "clipp1d"
    shutil.copytree(repo / "src/clipp1d", package,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(Path(__file__).with_name("legacy_chain_api.py"), package / "api.py")
    (package / "__init__.py").write_text(
        '"""Synthetic historical package for CPU regression tests only."""\n'
        '__version__ = "0.4.1"\nfrom .api import fit\n')
    env = dict(os.environ, PYTHONPATH=str(source_root), PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-B", "-c",
               "import json; from clipp1d.api import source_provenance; "
               "print(json.dumps(source_provenance(), sort_keys=True))"]
    completed = subprocess.run(command, env=env, check=True, capture_output=True, text=True, timeout=30,
                               cwd=source_root)
    return json.loads(completed.stdout)
