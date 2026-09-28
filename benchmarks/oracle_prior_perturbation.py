"""Evaluator-only original-prior truth-partition refits; never a fitting input.

Run after frozen arm outputs exist. These diagnostics cannot enter the proposal
bank or overwrite any public fit. 'Oracle' means given memberships, not a proved
global optimum of the clustering model.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_prior_perturbation import case_metrics, rows  # noqa: E402
from prior_perturbation import array_record, prepare_model, sha, write_json  # noqa: E402
from clipp1d.cuda.ancestry import refit_identity  # noqa: E402
from clipp1d.cuda.partition import refit_labels  # noqa: E402
from clipp1d.cuda.policy import QualificationError  # noqa: E402
from clipp1d.cuda_api import _validate_refit, require_cuda  # noqa: E402
from clipp1d.cuda.policy import CudaPolicy  # noqa: E402


def diagnose(dataset, results, outdir):
    device = require_cuda('cuda:0')
    torch.set_num_threads(1)
    outdir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((dataset/'EVALUATION_MANIFEST.json').read_text())
    records = []
    for case in manifest['cases']:
        path = results/case['case_id']
        complete = (path/'RESULT.json').is_file()
        metrics = case_metrics(case, dataset, results)['metrics'] if complete else None
        # Failure, or worsening relative to B on either primary accuracy measure.
        reasons = ['unpublished_case'] if not complete else [
            f'{arm}_worse_{measure}' for arm in ('P', 'D') for measure in ('ari', 'ccf_mae')
            if (metrics[arm][measure] < metrics['B'][measure] if measure == 'ari'
                else metrics[arm][measure] > metrics['B'][measure])]
        if not reasons:
            continue
        input_file, truth_file = dataset/case['input_file'], dataset/case['truth_file']
        assert sha(input_file) == case['input_sha256'] and sha(truth_file) == case['truth_sha256']
        truth = {r['mutation_id']: int(r['cluster_id']) for r in rows(truth_file)}
        _, model = prepare_model(input_file, device)
        labels = torch.tensor([truth[mid] for mid in model.mutation_ids], device=device)
        row = dict(case_id=case['case_id'], reasons=reasons, input_sha256=sha(input_file),
                   truth_sha256=sha(truth_file), status='unresolved', evaluator_only=True)
        try:
            fitted = refit_labels(model, labels)
            _validate_refit(model, fitted, CudaPolicy())
            row.update(status='qualified', refit=refit_identity(fitted),
                arrays=array_record(outdir/f"{case['case_id']}.npz", memberships=fitted.labels,
                                    phi=fitted.phi, centers=fitted.centers, gap=fitted.scalar.gap))
            lower, upper = float(fitted.score-2*fitted.gap), float(fitted.score)
            if complete:
                comparisons = {}
                for arm in ('P', 'D'):
                    search = json.loads((path/f'{arm}-search/SEARCH.json').read_text())
                    candidate = search['selected']
                    score, gap = candidate['score'], candidate['refit_gap']
                    delta = max(1e-10, 128*torch.finfo(torch.float64).eps*(1+abs(score)+abs(upper)))
                    comparisons[arm] = dict(truth_lower=lower, truth_upper=upper,
                        selected_lower=score-2*gap, selected_upper=score, numerical_allowance=delta,
                        interpretation='missed_lower_score_truth_partition' if upper < score-2*gap-delta else
                        'criterion_prefers_selected_to_truth_partition' if score < lower-delta else
                        'score_intervals_overlap')
                row['comparisons'] = comparisons
        except QualificationError as error:
            row.update(error=str(error), diagnostics=error.diagnostics)
        records.append(row)
        del model
    write_json(outdir/'ORACLES.json', dict(records=records, input_manifest_sha256=sha(dataset/'EVALUATION_MANIFEST.json'),
        proposal_generation_used_truth=False, diagnostic_refits_use_truth=True,
        scope='failed publication or P/D accuracy worsened relative to B; no feedback to fitting'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'results', 'outdir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.dataset, args.results, args.outdir)


if __name__ == '__main__':
    main()
