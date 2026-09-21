"""Fresh-process CPU timings and graph-state accounting; no speedup inference."""

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clipp1d import fit  # noqa: E402
from clipp1d.report import write_json  # noqa: E402
from simulate import simulate  # noqa: E402


def worker(outdir, size, seed):
    source, _ = simulate(outdir / "input", size, seed)
    started = perf_counter()
    result = fit(source, outdir / "fit")
    chain = result.frozen_chain
    record = {"mutations": size, "edges": chain.weights.size,
              "graph_array_bytes": chain.order.nbytes + chain.inverse_order.nbytes + chain.weights.nbytes,
              "elapsed_seconds": perf_counter() - started,
              "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "pilot_seconds": result.provenance["pilot_seconds"],
              "solver_seconds": sum(r.get("solver_seconds", 0) for r in result.search_diagnostics["path"]),
              "refit_seconds": sum(r.get("refit_seconds", 0) for r in result.search_diagnostics["path"]),
              "selected_clusters": len(result.cluster_centers), "selection_score": result.selection_score,
              "source_sha256": result.provenance["source_sha256"],
              "search": result.search_diagnostics, "backend": "cpu", "dtype": "float64"}
    write_json(outdir / "benchmark.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.outdir, args.worker, args.seed)
        return
    if args.outdir.exists():
        raise FileExistsError("Use a new benchmark directory")
    args.outdir.mkdir(parents=True)
    records = []
    for size in args.sizes:
        directory = args.outdir / f"n{size}"
        subprocess.run([sys.executable, __file__, "--worker", str(size), "--seed", str(args.seed),
                        "--outdir", str(directory)], check=True)
        records.append(json.loads((directory / "benchmark.json").read_text()))
    write_json(args.outdir / "scaling.json", {"schema": "clipp1d.scaling.v1", "results": records,
                                               "qualification": "synthetic CPU only; no paired speedup claim"})


if __name__ == "__main__":
    main()
