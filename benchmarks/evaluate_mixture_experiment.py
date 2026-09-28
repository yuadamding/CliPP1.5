"""Retrospective paired development screen; never a production adoption gate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, f1_score

from run_mixture_experiment import sha, write_json
from mixture_artifacts import ArtifactPaths
from clipp1d.io import read_tumor


def read(path):
    return json.loads(Path(path).read_text())


def concordance(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float(np.array_equal(x, y))
    return float(2*np.mean((x-x.mean())*(y-y.mean())) /
                 (x.var()+y.var()+(x.mean()-y.mean())**2))


def score(case_id, cohort, population, method, truth, table, ccf_column):
    ids = list(table.index)
    expected = truth.loc[ids]
    if not np.isfinite(expected.true_ccf).all() or expected.true_cluster.isna().any():
        raise ValueError('This development panel requires complete, unambiguous truth')
    centers = table.groupby('cluster_label')[ccf_column].first()
    members = table.groupby('cluster_label').apply(lambda g: min(g.index), include_groups=False)
    clonal = min(centers.index, key=lambda k: (abs(centers[k]-1), members[k]))
    true_smf = float((expected.true_ccf < 1-1e-12).mean())
    estimated_smf = float((table.cluster_label != clonal).mean())
    return dict(case_id=case_id, cohort=cohort, population=population, method=method,
                n=len(ids), true_k=expected.true_cluster.nunique(), selected_k=centers.size,
                ari=adjusted_rand_score(expected.true_cluster, table.cluster_label),
                ccf_mae=float(np.abs(expected.true_ccf-table[ccf_column]).mean()),
                true_smf=true_smf, estimated_smf=estimated_smf,
                smf_abs_error=abs(true_smf-estimated_smf))


def summarize(frame):
    single = frame.true_k == 1
    multi = ~single
    return dict(cases=len(frame), mean_ari=float(frame.ari.mean()),
                multi_mean_ari=float(frame[multi].ari.mean()) if multi.any() else None,
                mean_ccf_mae=float(frame.ccf_mae.mean()),
                smf_mae=float(frame.smf_abs_error.mean()),
                smf_ccc=concordance(frame.true_smf, frame.estimated_smf),
                single_cases=int(single.sum()),
                false_splits=int((single & (frame.selected_k > 1)).sum()),
                false_collapses=int((multi & (frame.selected_k == 1)).sum()),
                correct_k=int((frame.selected_k == frame.true_k).sum()))


def nonregression_check(candidate, baseline, *, maximize):
    """An undefined endpoint is unassessed; it cannot supply a passing gate."""
    if candidate is None or baseline is None:
        return None
    if not np.isfinite(candidate) or not np.isfinite(baseline):
        raise ValueError('Nonfinite comparison metric')
    return bool(candidate >= baseline-1e-12 if maximize else candidate <= baseline+1e-12)


def checked_comparators(comparators, cohort, case_id, matched_ids, baseline_metrics):
    """A missing method or same-size, different population must not disappear."""
    panel = comparators[(comparators.cohort_id == cohort) &
                        (comparators.population_id == 'matched')]
    expected = set(panel.method_id)
    case = panel[panel.case_id == case_id]
    if ('clipp15_baseline' not in expected or len(expected) < 2 or
            set(case.method_id) != expected or case.method_id.duplicated().any()):
        raise ValueError('Incomplete or duplicate comparator inventory')
    digest = hashlib.sha256(json.dumps(sorted(matched_ids), ensure_ascii=True).encode()).hexdigest()
    if (len(set(matched_ids)) != len(matched_ids) or
            not (case.n_mutations == len(matched_ids)).all() or
            not (case.mutation_ids_sha256 == digest).all()):
        raise ValueError('Comparator mutation population mismatch')
    original = case[case.method_id == 'clipp15_baseline'].iloc[0]
    for key in ('true_k', 'selected_k', 'ari', 'ccf_mae', 'true_smf',
                'estimated_smf', 'smf_abs_error'):
        if not np.isfinite(original[key]) or abs(original[key]-baseline_metrics[key]) > 1e-10:
            raise ValueError('Recomputed baseline does not match frozen comparator metrics: '+key)
    return case[case.method_id != 'clipp15_baseline']


def evaluate(inference_file, evaluation_file, directories, output, artifact_map=None):
    resolve = ArtifactPaths(artifact_map)
    inference, evaluation = read(inference_file), read(evaluation_file)
    if sha(inference_file) != evaluation['inference_sha256']:
        raise ValueError('Inference manifest is not bound to this evaluation')
    cases = {c['case_id']: c for c in inference['cases']}
    outputs, source_inventory, failures = {}, None, []
    selection_policy = None
    for directory in map(Path, directories):
        binding = read(directory/'BINDING.json')
        complete = read(directory/'COMPLETE.json')
        if not complete['source_unchanged'] or not complete['manifest_unchanged']:
            raise ValueError('Incomplete or invalid development output')
        failures.extend(dict(parent=str(directory), **r) for r in complete['failures'])
        if source_inventory is None:
            source_inventory = binding['source_inventory']
            selection_policy = binding.get('selection_policy')
        if (source_inventory != binding['source_inventory'] or binding['policy'] != inference['policy'] or
                binding.get('selection_policy') != selection_policy):
            raise ValueError('Panel source/configuration mismatch')
        for result in complete['results']:
            name = result['case_id']
            if name in outputs:
                raise ValueError('Duplicate output case')
            outputs[name] = resolve(result['output'])
    if set(outputs) != set(cases) or set(cases) != {c['case_id'] for c in evaluation['cases']}:
        raise ValueError('Missing or unexpected cases; do not select a favorable intersection')
    repaired = {r['case_id'] for r in evaluation.get('seed_preparation_repair', [])}
    if any(r['case_id'] not in repaired or r['case_id'] not in outputs or
           r['error'] != "ValueError('Saved seed partition must match the entire retained population')"
           for r in failures):
        raise ValueError('Unreconciled parent failure; only declared seed preparation repairs are allowed')
    candidate_label = 'Guarded soft mixture experiment' if selection_policy else 'Soft mixture experiment'
    comparison_file = resolve(evaluation['comparator_metrics'])
    if sha(comparison_file) != evaluation['comparator_metrics_sha256']:
        raise ValueError('Changed comparator metrics')
    comparators = pd.read_csv(comparison_file, sep='\t')
    rows, multiplicity, bindings, diagnostic = [], [], {}, []
    matched_authorities = {}
    for entry in evaluation['cases']:
        name, cohort = entry['case_id'], entry['cohort']
        case = cases[name]
        for key in ('input', 'seed'):
            if sha(resolve(case[key+'_path'])) != case[key+'_sha256']:
                raise ValueError('Changed input or baseline partition')
        if sha(resolve(entry['truth_path'])) != entry['truth_sha256']:
            raise ValueError('Changed truth')
        truth = pd.read_csv(resolve(entry['truth_path']), sep='\t', dtype={'mutation_id': str})
        if entry['truth_format'] == 'cnfirst_generator':
            truth['mutation_id'] = ['chr'+str(c)+':'+str(p) for c, p in
                                    zip(truth.chromosome_index, truth.position)]
            truth = truth.rename(columns=dict(cluster_id='true_cluster', ccf='true_ccf',
                                             multiplicity='true_multiplicity'))
        truth = truth.set_index('mutation_id', verify_integrity=True)
        baseline = pd.read_csv(resolve(case['seed_path']), sep='\t', dtype={'mutation_id': str}).set_index('mutation_id', verify_integrity=True)
        baseline = baseline.rename(columns={'multiplicity_call': 'multiplicity'})
        receipt = read(outputs[name]/'EXPERIMENT.json')
        if (receipt['inputs'] != case or receipt['policy'] != inference['policy'] or
                receipt.get('selection_policy') != selection_policy):
            raise ValueError('Output was fitted on different input or configuration')
        for filename, digest in receipt['output_sha256'].items():
            if sha(outputs[name]/filename) != digest:
                raise ValueError('Changed candidate output')
            bindings[str(outputs[name]/filename)] = digest
        candidate = pd.read_csv(outputs[name]/'mixture_mutation_clusters.tsv', sep='\t',
                                dtype={'mutation_id': str}).set_index('mutation_id', verify_integrity=True)
        if set(baseline.index) != set(candidate.index):
            raise ValueError('Candidate lost retained mutations')
        matched_file = resolve(entry['matched_ids_path'])
        expected_sha = entry['matched_ids_sha256']
        if matched_file not in matched_authorities:
            if sha(matched_file) != expected_sha:
                raise ValueError('Changed matched evaluation population')
            matched_authorities[matched_file] = (expected_sha, read(matched_file))
        digest, population = matched_authorities[matched_file]
        if digest != expected_sha:
            raise ValueError('Conflicting matched population authorities')
        matched_ids = population[name]
        for population, ids in [('retained', sorted(baseline.index)), ('matched', matched_ids)]:
            if not set(ids) <= set(baseline.index):
                raise ValueError('Matched IDs not in the fitted population')
            rows.append(score(name, cohort, population, 'CliPP1.5 baseline', truth,
                              baseline.loc[ids], 'partition_ccf'))
            rows.append(score(name, cohort, population, candidate_label, truth,
                              candidate.loc[ids], 'mixture_ccf'))
        external = checked_comparators(comparators, cohort, name, matched_ids, rows[-2])
        for _, row in external.iterrows():
            if int(row.n_mutations) != len(matched_ids):
                raise ValueError('Comparator population size mismatch')
            rows.append(dict(case_id=name, cohort=cohort, population='matched', method=row.method,
                             **{k: row[k] for k in ('true_k', 'selected_k', 'ari', 'ccf_mae',
                                                   'true_smf', 'estimated_smf', 'smf_abs_error')},
                             n=len(matched_ids)))
        # Eligibility uses the supplied fitting CN, and the target is the original
        # simulator's integer multiplicity (not its fractional effective dosage).
        # Original mixed-state flags are reported separately when supplied by the
        # versioned truth preparation. This is a development diagnostic only.
        for mutation in read_tumor(resolve(case['input_path'])).retained:
            states = {(a, b) for _, a, b in mutation.states}
            if len(states) != 1 or states == {(1, 1)}:
                continue
            mid = mutation.mutation_id
            actual = truth.loc[mid, 'true_multiplicity']
            if not np.isfinite(actual):
                raise ValueError('Missing CNA truth in development panel')
            for method, table in [('CliPP1.5 baseline', baseline), (candidate_label, candidate)]:
                call = table.loc[mid, 'multiplicity']
                multiplicity.append(dict(cohort=cohort, case_id=name, mutation_id=mid,
                                         method=method, truth=int(actual), call=int(call),
                                         original_mixed_cn=int(truth.loc[mid, 'mixed_cn'])
                                         if 'mixed_cn' in truth else 0))
        diagnostic.append(dict(case_id=name, cohort=cohort, adaptive=receipt['adaptive'],
                               selected_status=receipt['status'], seconds=receipt['elapsed_seconds'],
                               budget_exhausted_candidates=sum(r['status'] != 'em_fixed_point' for r in receipt['candidates'])))
    for path, (digest, _) in matched_authorities.items():
        if sha(path) != digest:
            raise ValueError('Matched population changed during evaluation')
    frame = pd.DataFrame(rows)
    summary = {f'{cohort}/{population}/{method}': summarize(g)
               for (cohort, population, method), g in frame.groupby(['cohort', 'population', 'method'])}
    mult = pd.DataFrame(multiplicity, columns=['cohort', 'case_id', 'mutation_id',
        'method', 'truth', 'call', 'original_mixed_cn'])
    multi_summary = {}
    for (cohort, method), group in mult.groupby(['cohort', 'method']):
        labels = sorted(set(mult[mult.cohort == cohort].truth) | set(mult[mult.cohort == cohort].call))
        multi_summary[f'{cohort}/{method}'] = dict(eligible=len(group), coverage=1.,
            macro_f1=f1_score(group.truth, group.call, labels=labels, average='macro', zero_division=0),
            micro_f1=f1_score(group.truth, group.call, labels=labels, average='micro', zero_division=0),
            weighted_f1=f1_score(group.truth, group.call, labels=labels, average='weighted', zero_division=0),
            classes=labels, per_class_f1=f1_score(group.truth, group.call, labels=labels, average=None, zero_division=0).tolist())
    for cohort in sorted(frame.cohort.unique()):
        for method in ('CliPP1.5 baseline', candidate_label):
            multi_summary.setdefault(f'{cohort}/{method}', dict(eligible=0, coverage=None,
                macro_f1=None, micro_f1=None, weighted_f1=None, classes=[], per_class_f1=[],
                status='no_eligible_CNA_loci'))
    checks = {}
    for cohort in sorted(frame.cohort.unique()):
        for population in ('retained', 'matched'):
            baseline = summary[f'{cohort}/{population}/CliPP1.5 baseline']
            candidate = summary[f'{cohort}/{population}/{candidate_label}']
            checks[f'{cohort}/{population}'] = {
                k: nonregression_check(candidate[k], baseline[k],
                    maximize=k in ('mean_ari', 'multi_mean_ari', 'smf_ccc'))
                for k in ('mean_ari', 'multi_mean_ari', 'smf_ccc', 'smf_mae', 'mean_ccf_mae',
                          'false_splits', 'false_collapses')}
    multiplicity_checks = {}
    for cohort in sorted(frame.cohort.unique()):
        baseline = multi_summary[f'{cohort}/CliPP1.5 baseline']
        candidate = multi_summary[f'{cohort}/{candidate_label}']
        multiplicity_checks[cohort] = {k: nonregression_check(candidate[k], baseline[k], maximize=True)
                                       for k in ('macro_f1', 'coverage')}
    headline_pass = all(all(c.values()) for c in checks.values())
    multiplicity_pass = all(all(c.values()) for c in multiplicity_checks.values())
    output = Path(output)
    output.mkdir()
    frame.to_csv(output/'per_case.tsv', sep='\t', index=False)
    mult.to_csv(output/'multiplicity.tsv', sep='\t', index=False)
    pd.DataFrame(diagnostic).to_csv(output/'diagnostics.tsv', sep='\t', index=False)
    write_json(output/'SUMMARY.json', dict(summary=summary, supplied_cn_multiplicity=multi_summary,
        development_nonregression_checks=checks,
        development_supplied_cn_multiplicity_checks=multiplicity_checks,
        development_unassessed_checks=[f'{group}/{key}'
            for group, values in {**checks, **multiplicity_checks}.items()
            for key, value in values.items() if value is None],
        development_headline_all_pass=headline_pass,
        development_all_pass=headline_pass and multiplicity_pass,
        multiplicity_truth_target='original simulator integer multiplicity on supplied-CN non-1/1 loci',
        mixed_cn_truth_schema=evaluation.get('mixed_cn_truth_schema', 'legacy or undeclared'),
        full_cohort_accepted=False, held_out_accepted=False, cuda_qualified=False, production_adopted=False,
        scope=evaluation['scope'], evaluator_sha256=sha(__file__),
        recovered_seed_preparation_failures=failures, selection_policy=selection_policy,
        inference_manifest_sha256=sha(inference_file), evaluation_manifest_sha256=sha(evaluation_file),
        source_inventory_sha256=hashlib.sha256(json.dumps(source_inventory, sort_keys=True).encode()).hexdigest(),
        output_bindings=bindings, artifact_map=resolve.mapping,
        artifact_map_sha256=sha(artifact_map) if artifact_map is not None else None))
    print(json.dumps(dict(summary=summary, development_nonregression_checks=checks), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inference-manifest', required=True)
    p.add_argument('--evaluation-manifest', required=True)
    p.add_argument('--result-dirs', nargs='+', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--artifact-map', help='Verified tree relocation; original receipts remain unchanged')
    args = p.parse_args()
    evaluate(args.inference_manifest, args.evaluation_manifest, args.result_dirs, args.outdir,
             args.artifact_map)


if __name__ == '__main__':
    main()
