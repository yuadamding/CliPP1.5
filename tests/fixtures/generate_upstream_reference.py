"""Explicit fixture maintenance only; never imported by the tests or package.

Run with ml1: python tests/fixtures/generate_upstream_reference.py /data/CliPP2/CliPP2
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

PIN = "77525a6875e698f6834cb73e6d2a1c27af2af8bf"
root = Path(sys.argv[1]).resolve()
assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == PIN
assert not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
sys.path.insert(0, str(root.parent))
from CliPP2.io.tumor_txt import load_tumor_txt  # noqa: E402
from CliPP2.core.objective import compile_observed_model, observed_terms_numpy  # noqa: E402

fixtures = Path(__file__).resolve().parent
data = load_tumor_txt(fixtures / "mixed_cn.tsv")
model = compile_observed_model(data, eps=1e-6)
phi = np.array([np.full(len(data.mutation_ids), 1e-6),
                np.full(len(data.mutation_ids), 1e-5),
                model.upper[:, 0] * .35, model.upper[:, 0] * .85, model.upper[:, 0]])
terms = [observed_terms_numpy(model, x[:, None], eps=1e-6) for x in phi]
target = fixtures / "upstream_reference.npz"
np.savez(target, phi=phi, upper=model.upper[:, 0], slope=model.slope[:, 0, :],
         loss=np.array([t.loss[:, 0] for t in terms]),
         gradient=np.array([t.gradient[:, 0] for t in terms]),
         curvature=np.array([t.hessian_upper[:, 0] for t in terms]),
         posterior=np.array([t.posterior[:, 0, :] for t in terms]))
(fixtures / "upstream_reference.json").write_text(json.dumps({
    "commit": PIN, "npz_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    "input_sha256": hashlib.sha256((fixtures / "mixed_cn.tsv").read_bytes()).hexdigest(),
    "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                       ("core/objective.py", "io/tumor_txt.py", "core/scalar.py", "core/bic.py")},
    "python": sys.executable, "numpy": np.__version__,
}, indent=2) + "\n")
