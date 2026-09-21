"""Isolate scalar and local-proposal work from full-fit solver changes."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clipp1d.api import source_provenance  # noqa: E402
from clipp1d.model import compile_model, evaluate  # noqa: E402
from clipp1d.io import read_tumor  # noqa: E402
from clipp1d.report import write_json  # noqa: E402
from clipp1d.scalar import compute_pilot, minimize_block  # noqa: E402
from clipp1d.solver import local_interval_delta, objective  # noqa: E402
from clipp1d.types import CountModel  # noqa: E402
from simulate import simulate  # noqa: E402

BASELINE = "4e9fcf62eb57908af3e15c45ddbd7bf035e10ad0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000])
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source = subprocess.run(["git", "show", f"{BASELINE}:src/clipp1d/scalar.py"], cwd=root,
                            capture_output=True, check=True).stdout
    # Execute only this repository's exact initial scalar module; common likelihood
    # and type definitions isolate scalar-search work from unrelated solver changes.
    old = types.ModuleType("clipp1d._baseline_scalar")
    old.__package__ = "clipp1d"
    exec(compile(source, f"git:{BASELINE}:scalar.py", "exec"), old.__dict__)
    setup = dict(provenance=source_provenance(), baseline_commit=BASELINE,
                 baseline_scalar_sha256=hashlib.sha256(source).hexdigest(),
                 benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 scope="isolated scalar/local kernels with common current likelihood, not old full-fit qualification")
    write_json(args.outdir / "setup.json", setup)
    for n in args.sizes:
        for scenario in ("easy", "mixture"):
            path, _ = simulate(args.outdir / f"{scenario}-{n}", n, 17, ambiguous=scenario == "mixture")
            model = compile_model(read_tumor(path))
            times, pilots = {}, []
            for name, method in (("initial", old.compute_pilot), ("revised", compute_pilot)):
                begin = perf_counter()
                pilots.append(method(model))
                times[name] = perf_counter() - begin
            write_json(args.outdir / f"pilot-{scenario}-{n}.json", dict(mutations=n, scenario=scenario,
                       seconds=times, maximum_loss_difference=float(np.max(abs(pilots[0].losses - pilots[1].losses))),
                       maximum_ccf_difference=float(np.max(abs(pilots[0].phi - pilots[1].phi))),
                       input_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print(f"pilots {scenario} M={n}: {times}", flush=True)
        model = CountModel(tuple(map(str, range(n))), np.zeros(n), np.full(n, 100), np.full(n, 1e-6),
                           np.ones(n), np.linspace(.1, .5, n)[:, None], np.zeros((n, 1)), np.ones((n, 1), dtype=bool), 1e-6)
        times, results = {}, []
        for name, method in (("initial", old.minimize_block), ("revised", minimize_block)):
            begin = perf_counter()
            results.append(method(model))
            times[name] = perf_counter() - begin
        write_json(args.outdir / f"lazy-{n}.json", dict(mutations=n, seconds=times,
                   both_qualified=all(r.qualified for r in results),
                   loss_difference=abs(results[0].attained_loss - results[1].attained_loss),
                   revised_evaluations=results[1].evaluations, revised_bound_evaluations=results[1].bound_evaluations))
        x, caps = np.full(n, .2), np.full(n - 1, .5)
        baseline_loss = objective(model, x, caps)
        block = model.subset([n // 2])
        old_loss = evaluate(block, x[n // 2:n // 2 + 1]).loss
        proposals = np.linspace(.19, .21, 100)
        begin = perf_counter()
        full = []
        for value in proposals:
            trial = x.copy()
            trial[n // 2] = value
            full.append(objective(model, trial, caps) - baseline_loss)
        full_seconds = perf_counter() - begin
        begin = perf_counter()
        local = [local_interval_delta(block, x, caps, n // 2, n // 2 + 1, v, old_loss) for v in proposals]
        local_seconds = perf_counter() - begin
        write_json(args.outdir / f"local-{n}.json", dict(mutations=n, proposals=len(proposals),
                   full_seconds=full_seconds, local_seconds=local_seconds,
                   maximum_delta_difference=float(np.max(abs(np.array(full) - local)))))
        print(f"lazy/local M={n}: measured", flush=True)


if __name__ == "__main__":
    main()
