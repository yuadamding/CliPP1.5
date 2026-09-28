"""Bound, no-clobber component experiment from saved complete-graph partitions.

CUDA is required by default. --cpu-reference is an explicit development-only
adapter; neither route launches a new raw fusion fit or promotes this estimator.
The fit manifest has no truth path. Evaluate its outputs separately afterward.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clipp1d.cuda.mixture import MixturePolicy, fit_mixture, fit_mixture_reference
from clipp1d.cuda.model import TensorModel
from clipp1d.io import read_tumor
from clipp1d.model import compile_model


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_case(case):
    if set(case) != {"case_id", "input_path", "input_sha256", "seed_path", "seed_sha256"}:
        raise ValueError("Inference cases accept only ID and bound input/partition seeds")
    for kind in ("input", "seed"):
        if sha(case[f"{kind}_path"]) != case[f"{kind}_sha256"]:
            raise ValueError(f"Changed {kind} artifact for {case['case_id']}")


def prepare_model(case, device):
    validate_case(case)
    host = compile_model(read_tumor(case['input_path']))
    host = host.subset(np.argsort(host.mutation_ids, kind="stable"))
    table = pd.read_csv(case['seed_path'], sep="\t", dtype={'mutation_id': str})
    if (table.mutation_id.duplicated().any() or
            set(table.mutation_id) != set(host.mutation_ids)):
        raise ValueError("Saved seed partition must match the entire retained population")
    table = table.set_index('mutation_id').loc[list(host.mutation_ids)]
    if table.cluster_label.isna().any() or not np.isfinite(table.partition_ccf).all():
        raise ValueError("Seed labels and centers must be present and finite")
    if (table.partition_ccf.to_numpy() < host.lower).any() or \
            (table.partition_ccf.to_numpy() > host.upper).any():
        raise ValueError("Seed partition violates original CCF bounds")
    if table.groupby('cluster_label').partition_ccf.nunique().max() != 1:
        raise ValueError("Each saved partition group must have one center")
    centers = table.groupby('cluster_label').partition_ccf.first().to_numpy()
    model = TensorModel.from_host(host, device, compiled=device.startswith("cuda"))
    seeds = torch.tensor(centers, device=device, dtype=torch.float64)
    return model, seeds


def run_case(case, policy, destination, device, cpu_reference):
    started = time.perf_counter()
    model, seeds = prepare_model(case, device)
    fit = fit_mixture_reference if cpu_reference else fit_mixture
    result = fit(model, policy, seed_centers=seeds)
    public = result.public_arrays(model.mutation_ids)
    phi = public['ccf']
    if not bool(torch.isfinite(phi).all() & (phi >= model.lower).all() &
                (phi <= model.upper).all() & torch.isfinite(result.posterior).all() &
                (result.posterior >= 0).all()):
        raise ArithmeticError("Invalid experimental output")
    if not torch.allclose(result.posterior.sum((1, 2)), torch.ones_like(phi), rtol=0, atol=1e-10):
        raise ArithmeticError("Invalid posterior normalization")
    table = pd.DataFrame(dict(mutation_id=model.mutation_ids,
                               cluster_label=public['labels'].cpu().numpy(),
                               mixture_ccf=phi.cpu().numpy(),
                               multiplicity=public['multiplicity'].cpu().numpy()))
    centers = pd.DataFrame(dict(cluster_label=np.arange(len(public['centers'])),
                                 mixture_ccf=public['centers'].cpu().numpy(),
                                 latent_component=public['latent_component'].cpu().numpy()))
    centers['n_mutations'] = table.groupby('cluster_label').size().reindex(centers.cluster_label).to_numpy()
    destination.mkdir()
    table.to_csv(destination / 'mixture_mutation_clusters.tsv', sep='\t', index=False,
                 float_format='%.17g')
    centers.to_csv(destination / 'mixture_cluster_centers.tsv', sep='\t', index=False,
                   float_format='%.17g')
    receipt = dict(result.metadata(), case_id=case['case_id'], inputs=case,
                   centers=result.centers.tolist(), weights=result.weights.tolist(),
                   enrichment=result.enrichment.tolist(),
                   elapsed_seconds=time.perf_counter()-started,
                   mutation_ids_sha256=hashlib.sha256(json.dumps(model.mutation_ids).encode()).hexdigest(),
                   device=str(model.device),
                   gpu_name=torch.cuda.get_device_name(model.device) if model.device.type == 'cuda' else None,
                   utc=datetime.now(timezone.utc).isoformat(),
                   output_sha256={name: sha(destination / name) for name in
                                  ('mixture_mutation_clusters.tsv', 'mixture_cluster_centers.tsv')})
    # Detect source changes during a fit before publishing a case receipt.
    validate_case(case)
    write_json(destination / 'EXPERIMENT.json', receipt)
    return dict(case_id=case['case_id'], score=result.score, selected_status=result.status,
                output=str(destination), elapsed_seconds=receipt['elapsed_seconds'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-reference', action='store_true')
    args = parser.parse_args(argv)
    if args.device.startswith('cpu') != args.cpu_reference:
        parser.error('CPU diagnostics require BOTH --device cpu and --cpu-reference')
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'clipp1d.mixture_experiment_inputs.v1':
        raise ValueError('Unrecognized inference manifest')
    policy = MixturePolicy(**manifest['policy'])
    cases = manifest['cases']
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Expected a nonempty unique case inventory')
    for case in cases:
        validate_case(case)
    root = Path(args.outdir).resolve()
    root.mkdir()  # Never resume into or overwrite existing results.
    source = Path(__file__).resolve().parents[1]
    inventory = {str(path.relative_to(source)): sha(path)
                 for path in sorted((source / 'src/clipp1d').rglob('*.py'))}
    inventory[str(Path(__file__).resolve().relative_to(source))] = sha(__file__)
    binding = dict(schema='clipp1d.mixture_experiment_binding.v1', manifest=str(manifest_path),
                   manifest_sha256=sha(manifest_path), source_inventory=inventory,
                   policy=asdict(policy), execution='cpu_component_reference' if args.cpu_reference else 'compiled_cuda',
                   python=sys.executable, torch=torch.__version__, numpy=np.__version__,
                   cuda=torch.version.cuda, started_utc=datetime.now(timezone.utc).isoformat(),
                   promotion_authorized=False, full_raw_fits=False)
    write_json(root / 'BINDING.json', binding)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    results, failures = [], []
    for index, case in enumerate(cases):
        try:
            row = run_case(case, policy, root / f'{index:05}', args.device, args.cpu_reference)
            results.append(row)
            print(json.dumps(dict(done=len(results), total=len(cases), **row)), flush=True)
        except Exception as exc:
            failures.append(dict(case_id=case['case_id'], error=repr(exc)))
            print(json.dumps(failures[-1]), flush=True)
    unchanged = all(sha(source / path) == value for path, value in inventory.items())
    manifest_unchanged = sha(manifest_path) == binding['manifest_sha256']
    write_json(root / 'COMPLETE.json', dict(results=results, failures=failures,
                                           source_unchanged=unchanged,
                                           manifest_unchanged=manifest_unchanged,
                                           numerical_qualification=False,
                                           accuracy_acceptance=False,
                                           utc=datetime.now(timezone.utc).isoformat()))
    return 0 if not failures and unchanged and manifest_unchanged else 1


if __name__ == '__main__':
    raise SystemExit(main())
