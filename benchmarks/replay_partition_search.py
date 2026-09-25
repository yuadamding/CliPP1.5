"""Replay frozen partitions; never invoke a new production fusion fit.

The portable bundle supplies canonical TSV bytes, published labels/centers,
independent expected assignments, truth for evaluation only, and matched IDs.
CUDA uses compiled production kernels. --device cpu is explicit reference
evidence, not CUDA qualification or a production fallback.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, f1_score
from clipp1d.api import source_provenance
from clipp1d.io import read_tumor
from clipp1d.model import compile_model
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.partition import canonical_labels, grouping, refit_labels
from clipp1d.cuda.refinement import PartitionSearchPolicy, reassign_fixed_centers, refine_memberships


def metrics(model, labels, centers, case, mask):
    phi = centers[labels]
    calls = model.terms(phi)[3].argmax(-1) + 1
    y = labels.cpu().numpy()[mask]
    estimates = phi.cpu().numpy()[mask]
    actual = np.asarray(case['true_ccf'])[mask]
    true_labels = np.asarray(case['true_labels'])[mask]
    occupied = np.unique(y)
    c = centers.cpu().numpy()
    clonal = min(occupied, key=lambda k: (abs(c[k]-1), int(np.flatnonzero(y == k)[0])))
    cna = np.asarray(case['cna'])[mask]
    observed_m = calls.cpu().numpy()[mask]
    true_m = np.asarray(case['true_multiplicity'])[mask]
    return dict(n=int(mask.sum()), ari=float(adjusted_rand_score(true_labels, y)),
                ccf_mae=float(np.abs(estimates-actual).mean()), true_k=len(np.unique(true_labels)),
                k=len(occupied), true_smf=float((actual < 1).mean()), smf=float((y != clonal).mean()),
                cna_macro_f1=float(f1_score(true_m[cna], observed_m[cna], labels=[1, 2, 3, 4], average='macro', zero_division=0)),
                cna_true=true_m[cna].tolist(), cna_pred=observed_m[cna].tolist())


def summarize(rows):
    answer = []
    for population in ('native', 'matched'):
        for stage in sorted({r['stage'] for r in rows}):
            selected = [r for r in rows if r['population'] == population and r['stage'] == stage]
            if not selected:
                continue
            actual = np.array([r['true_smf'] for r in selected])
            estimated = np.array([r['smf'] for r in selected])
            denominator = actual.var() + estimated.var() + (actual.mean()-estimated.mean())**2
            ccc = float(2 * np.mean((actual-actual.mean())*(estimated-estimated.mean())) / denominator) if denominator else 1.
            truth_m = sum((r['cna_true'] for r in selected), [])
            pred_m = sum((r['cna_pred'] for r in selected), [])
            answer.append(dict(population=population, stage=stage, cases=len(selected),
                               mutations=sum(r['n'] for r in selected),
                               mean_ari=float(np.mean([r['ari'] for r in selected])),
                               mean_ccf_mae=float(np.mean([r['ccf_mae'] for r in selected])),
                               mean_k=float(np.mean([r['k'] for r in selected])),
                               correct_k=sum(r['k'] == r['true_k'] for r in selected),
                               overclustered=sum(r['k'] > r['true_k'] for r in selected),
                               underclustered=sum(r['k'] < r['true_k'] for r in selected),
                               smf_ccc=ccc, smf_mae=float(np.abs(actual-estimated).mean()),
                               cna_macro_f1=float(f1_score(truth_m, pred_m, labels=[1, 2, 3, 4], average='macro', zero_division=0))))
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--alternating-refit', action='store_true')
    parser.add_argument('--maximum-cases', type=int)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.bundle.read_bytes())
    device = torch.device(args.device)
    if device.type == 'cuda':
        from clipp1d.cuda_api import require_cuda
        device = require_cuda(args.device)
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type != 'cpu':
        raise ValueError('Only explicit CPU reference or CUDA execution is supported')
    torch.set_num_threads(1)
    records, comparisons, unresolved = [], [], []
    cases = data['cases'][:args.maximum_cases] if args.maximum_cases else data['cases']
    for case in cases:
        begin = perf_counter()
        path = args.outdir/(case['case_id']+'.tsv')
        assert hashlib.sha256(case['input_tsv'].encode()).hexdigest() == case['input_sha256']
        path.write_text(case['input_tsv'])
        host = compile_model(read_tumor(path))
        host = host.subset(np.argsort(host.mutation_ids, kind='stable'))
        assert list(host.mutation_ids) == case['mutation_ids']
        model = TensorModel.from_host(host, device, compiled=device.type == 'cuda')
        raw_labels = torch.tensor(case['published_labels'], device=device, dtype=torch.long)
        labels, order, sizes = canonical_labels(raw_labels)
        old_phi = torch.tensor(case['published_ccf'], device=device, dtype=torch.float64)
        centers = old_phi[order[sizes.cumsum(0)-sizes]]
        fixed = reassign_fixed_centers(model, labels, centers)
        expected = torch.tensor(case['expected_fixed_labels'], device=device, dtype=torch.long)
        expected = canonical_labels(expected)[0]
        mismatch = int((fixed.labels != expected).sum())
        stage_results = [('published', labels, centers), ('fixed_centers', fixed.labels, fixed.centers)]
        if args.alternating_refit:
            try:
                # Match the production four-round budget from the original seed;
                # do not silently add the preceding diagnostic sweep as round 0.
                initial = refit_labels(model, labels, incumbent_centers=centers)
                refined, rounds, status = refine_memberships(model, initial)
                stage_results.append(('alternating_refit', refined.labels, refined.centers))
                unresolved.append(dict(case_id=case['case_id'], status=status, rounds=rounds,
                                       score=float(refined.score), fixed_score=float(fixed.score)))
            except Exception as error:
                # Record and fail qualification at the end; never silently omit failures.
                unresolved.append(dict(case_id=case['case_id'], status='failure', error=repr(error)))
        raw_phi = torch.tensor(case['raw_ccf'], dtype=torch.float64, device=device)
        legacy = grouping(raw_phi, 2e-5)[2]
        generic = grouping(raw_phi, 2e-5, separate_clonal=False)[2]
        for population, mask in [('native', np.ones(model.n, dtype=bool)),
                                 ('matched', np.array(case['matched_mask'], dtype=bool))]:
            for stage, z, c in stage_results:
                records.append(dict(case_id=case['case_id'], population=population, stage=stage,
                                    **metrics(model, z, c, case, mask)))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        comparisons.append(dict(case_id=case['case_id'], expected_label_mismatches=mismatch,
                                grouping_variant_changed=not torch.equal(legacy, generic),
                                fixed_score=float(fixed.score), expected_score=case['expected_fixed_score'],
                                score_difference=float(fixed.score)-case['expected_fixed_score'],
                                fixed_status=fixed.status, moves=len(fixed.moves), seconds=perf_counter()-begin))
        print(f'{len(comparisons)}/{len(cases)} {case["case_id"]} label_mismatches={mismatch}', flush=True)
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), source=source_provenance(),
                   bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(), device=str(device),
                   execution='compiled_cuda_partition_replay' if device.type == 'cuda' else 'cpu_reference_partition_replay',
                   discovery_panel=True, new_fusion_fits=False,
                   alternating_seed='qualified_original_memberships',
                   search_policy=PartitionSearchPolicy().__dict__,
                   comparisons=comparisons, alternating=unresolved, summary=summarize(records),
                   exact_membership_matches=sum(r['expected_label_mismatches'] == 0 for r in comparisons),
                   largest_score_difference=max(abs(r['score_difference']) for r in comparisons),
                   runtime_median_seconds=float(np.median([r['seconds'] for r in comparisons])))
    if device.type == 'cuda':
        receipt.update(gpu_name=torch.cuda.get_device_name(device), peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    with (args.outdir/'REPLAY.json').open('x') as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
    with (args.outdir/'PER_CASE.json').open('x') as f:
        json.dump(records, f, indent=2, allow_nan=False)
    assert all(r['expected_label_mismatches'] == 0 for r in comparisons), 'Fixed-center replay differs; inspect per-case evidence'
    assert all(r['status'] != 'failure' for r in unresolved), 'Qualified refit failed; inspect evidence'
    print(json.dumps({k:receipt[k] for k in ('execution', 'exact_membership_matches', 'largest_score_difference', 'summary')}, indent=2))


if __name__ == '__main__':
    main()
