"""Pilot-only discovery and separate fixed-penalty prior/graph factorial."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prior_perturbation import (  # noqa: E402
    MAGNITUDES, array_record, auxiliary_model, deterministic_starts,
    keyed_directions, model_identity, pilot_proposals, prepare_model, sha,
    synchronized_time, write_json,
)
from clipp1d.cuda.experimental import solve_extra_start  # noqa: E402
from clipp1d.cuda.graph import build_graph  # noqa: E402
from clipp1d.cuda.partition import grouping, refit_labels  # noqa: E402
from clipp1d.cuda.policy import CudaPolicy, QualificationError  # noqa: E402
from clipp1d.cuda.refinement import PartitionSearchPolicy  # noqa: E402
from clipp1d.cuda.scalar import pilot  # noqa: E402
from clipp1d.cuda.solver import objective  # noqa: E402
from clipp1d.cuda_api import _tensor_sha256, require_cuda  # noqa: E402


def discovery_case(input_file, tumor_id, outdir, device='cuda:0'):
    device = require_cuda(device)
    torch.set_num_threads(1)
    outdir.mkdir(parents=True, exist_ok=False)
    _, model = prepare_model(input_file, device)
    original = pilot(model)
    original_record = array_record(outdir/'original.npz', phi=original.phi, loss=original.loss,
        alternative=original.alternative, alternative_loss=model.loss(original.alternative),
        support=torch.isfinite(model.log_prior).sum(-1), mutation_ids=np.asarray(model.mutation_ids))
    records = {}
    for eta in MAGNITUDES:
        _, records[str(eta)] = pilot_proposals(model, original.phi, tumor_id, outdir/f'eta-{eta}', eta)
    write_json(outdir/'DISCOVERY.json', dict(input_sha256=sha(input_file), original=original_record,
        results=records, discovery_only=True, fusion_solves=0, truth_used=False,
        well_gap_scope='finite original scalar-solver alternative candidate bank'))


def select_factorial(development_results, fitting_manifest, output):
    """Rank pilot ambiguity, split into equal thirds, hash-pick four per third."""
    cases = json.loads(fitting_manifest.read_text())['cases']
    ranked = []
    for case in cases:
        pilots = json.loads((development_results/case['case_id']/'P-pilots/PILOTS.json').read_text())
        qualified = [r['substantial_shift_fraction'] for r in pilots['records'] if r['status'] == 'qualified']
        if len(qualified) != 8:
            raise QualificationError('Factorial selection needs all eight prespecified pilot diagnostics')
        ambiguity = float(np.mean(qualified))
        digest = hashlib.sha256(f"2026092600:factorial:{case['case_id']}".encode()).hexdigest()
        ranked.append(dict(case, ambiguity=ambiguity, selection_hash=digest))
    ranked.sort(key=lambda c: (c['ambiguity'], c['selection_hash']))
    groups = np.array_split(np.arange(len(ranked)), 3)
    chosen = []
    for group, indices in enumerate(groups):
        candidates = sorted([ranked[int(i)] for i in indices], key=lambda c: c['selection_hash'])
        if len(candidates) < 4:
            raise ValueError('Fewer than four cases in an ambiguity tertile')
        chosen.extend(dict(case, ambiguity_group=group) for case in candidates[:4])
    write_json(output, dict(cases=chosen, selection_uses_accuracy=False,
                           ambiguity='mean fraction |auxiliary pilot-original pilot| >= .05 over eight directions'))
    return chosen


def factorial_case(input_file, tumor_id, baseline_root, outdir, device='cuda:0'):
    device = require_cuda(device)
    torch.set_num_threads(1)
    outdir.mkdir(parents=True, exist_ok=False)
    _, original = prepare_model(input_file, device)
    baseline = json.loads((baseline_root/'RESULT.json').read_text())
    if sha(input_file) != baseline['input_sha256'] or model_identity(original) != baseline['model_sha256']:
        raise ValueError('Factorial input/model differs from the frozen baseline')
    original_pilot = pilot(original)
    graph0 = build_graph(original_pilot.phi, original.mutation_ids)
    if (_tensor_sha256(graph0.weights) != baseline['graph_weights_sha256'] or
            float(graph0.gap_floor) != baseline['graph_recipe']['gap_floor'] or
            float(graph0.normalization) != baseline['graph_recipe']['normalization']):
        raise ValueError('Factorial original graph differs from baseline')
    raw_receipt = json.loads((baseline_root/'R/run.json').read_text())
    positive = [r['lambda_value'] for r in raw_receipt['search'] if r['lambda_value'] > 0]
    if not positive:
        raise QualificationError('No positive baseline penalty for the fixed-penalty factorial')
    selected = baseline['selected_baseline_lambda']
    penalty = selected if selected > 0 else min(positive, key=lambda x: (abs(x-baseline['lambda_reference']), x))
    lam = original.alt.new_tensor(penalty)
    # Identical ORIGINAL numerical starts in all 32 cells, independent of xi.
    starts = [(0, original_pilot.phi.clone())]+[(i, x) for i, x in deterministic_starts(original)
                                              if not torch.equal(x, original_pilot.phi)]
    noise = keyed_directions(tumor_id, original.mutation_ids, original.slope.shape[1])
    rows = []
    for index, xi in enumerate(noise):
        auxiliary = auxiliary_model(original, torch.tensor(xi, device=device), .03)
        try:
            perturbed_pilot = pilot(auxiliary)
        except QualificationError as error:
            rows.append(dict(direction=index, status='unresolved_auxiliary_pilot', error=str(error)))
            continue
        graph1 = build_graph(perturbed_pilot.phi, auxiliary.mutation_ids)
        for name, model, graph in [('S00', original, graph0), ('S10', auxiliary, graph0),
                                    ('S01', original, graph1), ('S11', auxiliary, graph1)]:
            for start_index, start in starts:
                began = synchronized_time(device)
                row = dict(direction=index, arm=name, start=start_index, penalty=penalty,
                    model_sha256=model_identity(model), graph_weights_sha256=_tensor_sha256(graph.weights),
                    status='unresolved')
                try:
                    solved = solve_extra_start(model, graph, lam, start)
                    solved.validate(model, graph, lam, CudaPolicy())
                    raw = solved.raw
                    row.update(qualified=raw.qualified, own_objective=float(raw.objective),
                        common_original_objective=float(objective(original, raw.x, graph0.weights*lam)),
                        diagnostics=raw.diagnostics)
                    if not raw.qualified:
                        raise QualificationError('Factorial raw fit unresolved')
                    labels = grouping(raw.x, CudaPolicy().fusion_tol,
                                      separate_clonal=PartitionSearchPolicy().separate_exact_one)[2]
                    fitted = refit_labels(model, labels)
                    row.update(status='qualified', arrays=array_record(outdir/f'{name}-{index}-{start_index}.npz',
                        raw_phi=raw.x, memberships=fitted.labels, refitted_phi=fitted.phi,
                        multiplicity=model.terms(fitted.phi)[3].argmax(-1)+1,
                        prior=model.log_prior, start=start))
                except QualificationError as error:
                    row.update(error=str(error))
                row['seconds'] = synchronized_time(device)-began
                rows.append(row)
        del graph1, auxiliary
    write_json(outdir/'FACTORIAL.json', dict(penalty=penalty, rows=rows, truth_used=False,
        comparing_different_objective_values_forbidden=True, own_objectives_are_diagnostics_only=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['discovery', 'factorial', 'select'])
    parser.add_argument('--input-file', type=Path)
    parser.add_argument('--tumor-id')
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if args.mode == 'select':
        select_factorial(args.baseline, args.manifest, args.outdir)
    elif args.mode == 'discovery':
        discovery_case(args.input_file, args.tumor_id, args.outdir)
    else:
        factorial_case(args.input_file, args.tumor_id, args.baseline, args.outdir)


if __name__ == '__main__':
    main()
