"""Portable frozen-partition birth replay on explicit CPU reference or allocated CUDA.

Never fits a new fusion path. Default single-cluster mode checks the original
discovery winners; experimental modes are reported separately. Truth is used
only for evaluation. CPU success is not CUDA or held-out qualification.
"""
import argparse
from dataclasses import replace, asdict
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
from clipp1d.cuda.partition import canonical_labels, refit_labels, partition_score
from clipp1d.cuda.refinement import PartitionSearchPolicy, refine_memberships
from clipp1d.cuda.birth import birth_candidates
from clipp1d.cuda.ancestry import membership_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--birth-mode', choices=('off', 'single_cluster', 'any_cluster'), default='single_cluster')
    parser.add_argument('--maximum-cases', type=int)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    bundle = json.loads(args.bundle.read_bytes())
    if bundle.get('schema') != 'clipp1d.birth.replay.v1':
        raise ValueError('Unexpected frozen replay schema')
    device = torch.device(args.device)
    if device.type == 'cuda':
        from clipp1d.cuda_api import require_cuda
        device = require_cuda(args.device)
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type != 'cpu':
        raise ValueError('Explicit CPU reference or allocated CUDA required')
    torch.set_num_threads(1)
    policy = PartitionSearchPolicy(birth_mode=args.birth_mode)
    rows, failures = [], []
    cases = bundle['cases'][:args.maximum_cases] if args.maximum_cases else bundle['cases']
    for case in cases:
        begin = perf_counter()
        text = case['input_tsv']
        if hashlib.sha256(text.encode()).hexdigest() != case['input_sha256']:
            raise ValueError('Frozen input hash mismatch')
        path = args.outdir/(case['case_id']+'.tsv')
        path.write_text(text)
        host = compile_model(read_tumor(path))
        host = host.subset(np.argsort(host.mutation_ids, kind='stable'))
        assert list(host.mutation_ids) == case['mutation_ids']
        model = TensorModel.from_host(host, device, compiled=device.type == 'cuda')
        labels, order, sizes = canonical_labels(torch.tensor(case['labels'], device=device, dtype=torch.long))
        phi = torch.tensor(case['ccf'], device=device, dtype=torch.float64)
        centers = phi[order[sizes.cumsum(0)-sizes]]
        score = float(partition_score(model.loss(phi).sum(), sizes))
        assert abs(score-case['score']) < 1e-7
        records, rounds = [], []
        selected_labels, selected_centers, selected_score = labels, centers, score
        status = 'not_eligible'
        if args.birth_mode == 'any_cluster' or (args.birth_mode == 'single_cluster' and sizes.numel() == 1):
            initial = refit_labels(model, labels)
            best = initial
            bank = []
            for candidate, failure in birth_candidates(model, initial, search_policy=policy):
                if failure is not None:
                    records.append(failure)
                    if failure['status'] == 'unresolved':
                        failures.append(dict(case_id=case['case_id'], **failure))
                    continue
                r = candidate.refit
                records.append(dict(proposal=candidate.proposal, labels=r.labels.cpu().tolist(),
                                    centers=r.centers.cpu().tolist(), membership_sha256=membership_hash(r.labels),
                                    score=float(r.score), gap=float(r.gap)))
                if float(r.score) < float(best.score):
                    best = r
                bank.append(r)
                bank = sorted(bank, key=lambda c: (float(c.score), c.centers.numel()))[:policy.birth_refine_seeds]
            for seed in bank:
                refined, history, result_status = refine_memberships(model, seed,
                    search_policy=replace(policy, max_rounds=policy.birth_max_rounds))
                rounds.append(dict(seed_score=float(seed.score), status=result_status, rounds=history))
                if result_status == 'unresolved':
                    failures.append(dict(case_id=case['case_id'], status=result_status))
                if float(refined.score) < float(best.score):
                    best = refined
            selected_labels, selected_centers, selected_score = best.labels, best.centers, float(best.score)
            status = 'evaluated'
        assert selected_score <= score + 1e-7
        expected = case.get('expected_birth') if args.birth_mode == 'single_cluster' else None
        mismatch = None
        if expected:
            mismatch = int((selected_labels.cpu().numpy() != np.array(expected['labels'])).sum())
            assert mismatch == 0, case['case_id']
            np.testing.assert_allclose(selected_centers.cpu(), expected['centers'], atol=1e-6, rtol=0.)
            assert abs(selected_score-expected['score']) < 1e-5
        elif args.birth_mode == 'single_cluster':
            assert torch.equal(labels, selected_labels) and torch.equal(centers, selected_centers)
        mask = np.asarray(case['matched_mask'], bool)
        y = selected_labels.cpu().numpy()[mask]
        c = selected_centers.cpu().numpy()
        estimates = c[y]
        truth = np.asarray(case['truth_ccf'])[mask]
        truth_labels = np.asarray(case['truth_labels'])[mask]
        clonal = min(np.unique(y), key=lambda k: (abs(c[k]-1), int(np.flatnonzero(y == k)[0])))
        calls = model.terms(selected_centers[selected_labels])[3].argmax(-1).cpu().numpy()[mask]+1
        cna = np.asarray(case['cna'], bool)[mask]
        true_m = np.asarray(case['truth_multiplicity'])[mask]
        row = dict(case_id=case['case_id'], status=status, n=int(mask.sum()), k=len(np.unique(y)),
                   true_k=len(np.unique(truth_labels)), score=selected_score, original_score=score,
                   labels=selected_labels.cpu().tolist(), centers=c.tolist(),
                   expected_label_mismatches=mismatch, candidates=records, refinement=rounds,
                   ari=float(adjusted_rand_score(truth_labels, y)), ccf_mae=float(np.abs(estimates-truth).mean()),
                   true_smf=float(np.mean(truth < 1-1e-12)), smf=float(np.mean(y != clonal)),
                   cna_true=true_m[cna].tolist(), cna_pred=calls[cna].tolist(),
                   seconds=perf_counter()-begin)
        rows.append(row)
        print(f'{len(rows)}/{len(cases)} {case["case_id"]} {status} K={row["k"]}', flush=True)
    x = np.array([r['true_smf'] for r in rows])
    y = np.array([r['smf'] for r in rows])
    denominator = x.var()+y.var()+(x.mean()-y.mean())**2
    ccc = float(2*np.mean((x-x.mean())*(y-y.mean()))/denominator) if denominator else 1.
    truth_m = sum((r['cna_true'] for r in rows), [])
    pred_m = sum((r['cna_pred'] for r in rows), [])
    receipt = dict(schema='clipp1d.birth.replay.result.v1', created_utc=datetime.now(timezone.utc).isoformat(),
        source=source_provenance(), device=str(device), compiled=device.type == 'cuda', new_fusion_fits=False,
        discovery_panel=True, policy=asdict(policy), bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        cases=len(rows), failures=failures, results=rows,
        summary=dict(mean_ari=float(np.mean([r['ari'] for r in rows])),
                     mean_ccf_mae=float(np.mean([r['ccf_mae'] for r in rows])), smf_ccc=ccc,
                     collapsed_multi=sum(r['true_k'] > 1 and r['k'] == 1 for r in rows),
                     single_false_splits=sum(r['true_k'] == 1 and r['k'] > 1 for r in rows),
                     correct_k=sum(r['true_k'] == r['k'] for r in rows),
                     cna_macro_f1=float(f1_score(truth_m, pred_m, labels=[1, 2, 3, 4], average='macro', zero_division=0))))
    if device.type == 'cuda':
        receipt.update(gpu_name=torch.cuda.get_device_name(device), peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    (args.outdir/'REPLAY.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    assert not failures, failures
    print(json.dumps(receipt['summary'], indent=2))


if __name__ == '__main__':
    main()
