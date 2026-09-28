"""Evaluator-only truth access and prespecified paired, stratified acceptance."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

EVALUATION_SEED = 2026092699


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def ari(truth, inferred):
    truth, inferred = np.asarray(truth), np.asarray(inferred)
    a, b = np.unique(truth, return_inverse=True)[1], np.unique(inferred, return_inverse=True)[1]
    n = len(a)
    if n < 2:
        return 1.
    table = np.zeros((a.max()+1, b.max()+1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    def choose(counts):
        return float(np.sum(counts*(counts-1)/2))
    observed, first, second = choose(table), choose(table.sum(0)), choose(table.sum(1))
    expected = first*second/(n*(n-1)/2)
    denominator = (first+second)/2-expected
    return 1. if denominator == 0 else (observed-expected)/denominator


def multiplicity_metrics(truth, calls, selected):
    truth, calls, selected = np.asarray(truth), np.asarray(calls), np.asarray(selected, bool)
    truth, calls = truth[selected], calls[selected]
    classes = {}
    for k in range(1, 5):
        tp = int(((truth == k) & (calls == k)).sum())
        fp = int(((truth != k) & (calls == k)).sum())
        fn = int(((truth == k) & (calls != k)).sum())
        classes[str(k)] = dict(tp=tp, fp=fp, fn=fn, support=int((truth == k).sum()),
            precision=None if tp+fp == 0 else tp/(tp+fp), recall=None if tp+fn == 0 else tp/(tp+fn),
            f1=0. if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn))
    return dict(n=len(truth), accuracy=None if not len(truth) else float((truth == calls).mean()),
                class_metrics=classes, macro_f1=float(np.mean([v['f1'] for v in classes.values()])))


def case_metrics(case, dataset, result_root):
    truth_path = dataset/case['truth_file']
    assert sha(truth_path) == case['truth_sha256']
    truth = {row['mutation_id']: row for row in rows(truth_path)}
    record = json.loads((result_root/case['case_id']/'RESULT.json').read_text())
    assert record['input_sha256'] == case['input_sha256'] and record['truth_used'] is False
    measurements = {}
    for arm in ('R', 'B', 'P', 'D'):
        root = result_root/case['case_id']/arm
        receipt = json.loads((root/'run.json').read_text())
        assert receipt['provenance']['source_sha256'] == record['source']['source_sha256']
        assert receipt['graph_sha256'] == record['graph_sha256']
        assert receipt['provenance']['model_sha256'] == record['model_sha256']
        calls = rows(root/('mutation_clusters.tsv' if arm == 'R' else 'partition_mutation_clusters.tsv'))
        assert len(calls) == len(truth) and {c['mutation_id'] for c in calls} == set(truth)
        joined = [truth[c['mutation_id']] for c in calls]
        labels = [int(c['cluster_label']) for c in calls]
        true_labels = [int(t['cluster_id']) for t in joined]
        truth_ccf = np.array([float(t['ccf']) for t in joined])
        # R records its raw primary and original-prior refit separately.
        field = 'refitted_ccf' if arm == 'R' else 'ccf'
        if arm != 'R' and field not in calls[0]:
            field = 'partition_ccf'
        predicted = np.array([float(c[field]) for c in calls])
        k, true_k = len(set(labels)), len(set(true_labels))
        measurement = dict(ari=ari(true_labels, labels), ccf_mae=float(np.mean(np.abs(predicted-truth_ccf))),
            ccf_rmse=float(np.mean((predicted-truth_ccf)**2)**.5), k=k, true_k=true_k,
            exact_k=k == true_k, undercluster=k < true_k, overcluster=k > true_k,
            false_single=k == 1 and true_k > 1, false_split=true_k == 1 and k > 1,
            n=len(labels), seconds=record['seconds'][arm], score=record['result_scores'][arm])
        if arm == 'R':
            raw = np.array([float(c['raw_ccf']) for c in calls])
            measurement['raw_ccf_mae'] = float(np.mean(np.abs(raw-truth_ccf)))
            measurement['raw_ccf_rmse'] = float(np.mean((raw-truth_ccf)**2)**.5)
            mult_rows = {row['mutation_id']: row for row in rows(root/'mutation_multiplicity.tsv')}
            mult = [int(mult_rows[c['mutation_id']]['refitted_multiplicity_call']) for c in calls]
        else:
            mult = [int(c.get('multiplicity', c.get('multiplicity_call'))) for c in calls]
        t_mult = [int(t['multiplicity']) for t in joined]
        major = np.array([int(t['major_cn']) for t in joined])
        minor = np.array([int(t['minor_cn']) for t in joined])
        measurement['multiplicity'] = dict(
            all=multiplicity_metrics(t_mult, mult, np.ones(len(mult), bool)),
            ambiguous=multiplicity_metrics(t_mult, mult, major > 1),
            cna_only=multiplicity_metrics(t_mult, mult, (major != 1) | (minor != 1)))
        # Bind returned public output hashes, not only result filenames.
        assert receipt['status'] == 'success' and receipt['table_sha256']
        for name, expected in receipt['table_sha256'].items():
            assert isinstance(expected, str) and len(expected) == 64
            assert sha(root/name) == expected
        measurements[arm] = measurement
    return dict(case_id=case['case_id'], source_sha256=record['source']['source_sha256'],
                cell=[case[k] for k in ('depth', 'purity', 'cna_rate', 'design_index')],
                metrics=measurements)


def summarize(measurements, expected_cases, *, correctness_passed=False, performance=None):
    """Pair tumors (not noise replicates); 108 design cells in confirmation."""
    grouped = defaultdict(list)
    for i, row in enumerate(measurements):
        grouped[tuple(row['cell'])].append(i)
    if not measurements:
        raise ValueError('No paired validated tumors')
    rng = np.random.default_rng(EVALUATION_SEED)
    # Sampling within each design cell preserves the stratification and all arms.
    resamples = np.concatenate([rng.choice(v, (10000, len(v)), replace=True) for _, v in sorted(grouped.items())], axis=1)
    means, paired = {}, {}
    fields = ('ari', 'ccf_mae', 'ccf_rmse', 'seconds', 'exact_k', 'undercluster', 'overcluster', 'false_single', 'false_split')
    for arm in ('R', 'B', 'P', 'D'):
        means[arm] = {key: float(np.mean([r['metrics'][arm][key] for r in measurements])) for key in fields}
        means[arm]['mutation_weighted_ccf_mae'] = float(np.average(
            [r['metrics'][arm]['ccf_mae'] for r in measurements], weights=[r['metrics'][arm]['n'] for r in measurements]))
        means[arm]['counts'] = {key: sum(int(r['metrics'][arm][key]) for r in measurements)
            for key in ('exact_k', 'undercluster', 'overcluster', 'false_single', 'false_split')}
        if 'multiplicity' in measurements[0]['metrics'][arm]:
            pooled = {}
            for subset in ('all', 'ambiguous', 'cna_only'):
                items = [r['metrics'][arm]['multiplicity'][subset] for r in measurements]
                classes = {}
                for k in ('1', '2', '3', '4'):
                    counts = {field: sum(item['class_metrics'][k][field] for item in items)
                              for field in ('tp', 'fp', 'fn', 'support')}
                    tp, fp, fn = (counts[field] for field in ('tp', 'fp', 'fn'))
                    classes[k] = dict(counts, precision=None if tp+fp == 0 else tp/(tp+fp),
                        recall=None if tp+fn == 0 else tp/(tp+fn),
                        f1=0. if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn))
                n = sum(item['n'] for item in items)
                pooled[subset] = dict(n=n, accuracy=None if not n else sum(c['tp'] for c in classes.values())/n,
                    class_metrics=classes, macro_f1=float(np.mean([c['f1'] for c in classes.values()])),
                    class_scope='fixed exact multiplicity classes 1,2,3,4; undefined class F1 set to zero')
            means[arm]['pooled_multiplicity'] = pooled
    for against in ('B', 'D'):
        delta = np.array([r['metrics']['P']['ari']-r['metrics'][against]['ari'] for r in measurements])
        bootstrap = delta[resamples].mean(axis=1)
        paired[against] = dict(mean_ari_delta=float(delta.mean()), ci95=np.quantile(bootstrap, [.025, .975]).tolist())
    baseline_mae = means['B']['ccf_mae']
    reduction = None if baseline_mae == 0 else 100*(1-means['P']['ccf_mae']/baseline_mae)
    scores = [r['metrics']['P']['score'] <= r['metrics']['B']['score']+
              max(1e-10, 128*np.finfo(float).eps*(1+abs(r['metrics']['B']['score']))) for r in measurements]
    complete = len(measurements) == expected_cases
    gates = dict(correctness=correctness_passed, complete_paired_coverage=complete,
        incumbent_preserved=all(scores), ari=paired['B']['mean_ari_delta'] >= .01 and paired['B']['ci95'][0] > 0,
        ccf=reduction is not None and reduction >= 2,
        specific_advantage_over_D=paired['D']['ci95'][0] > 0,
        exact_k=means['P']['exact_k'] >= means['B']['exact_k'],
        false_splits=means['P']['false_split'] <= means['B']['false_split'],
        execution_reliability=complete, runtime=None, memory=None)
    if performance is not None:
        gates['runtime'] = performance['median_total_runtime_ratio'] <= 2 and performance['p95_total_runtime_ratio'] <= 3
        gates['memory'] = performance['maximum_peak_memory_ratio'] <= 1.5
    return dict(paired_tumors=len(measurements), expected_tumors=expected_cases, means=means, paired=paired,
        mae_reduction_percent=reduction, bootstrap_seed=EVALUATION_SEED, bootstrap_resamples=10000,
        bootstrap_scope='tumors within depth/purity/CNA/cluster-design cells',
        interval_identifiable=all(len(v) > 1 for v in grouped.values()), gates=gates,
        adoption_supported=all(value is True for value in gates.values()) and all(len(v) > 1 for v in grouped.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--performance', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.dataset/'EVALUATION_MANIFEST.json').read_text())
    assert sha(args.dataset/'FIT_MANIFEST.json') == manifest['fitting_manifest_sha256']
    measurements, unavailable, publication_coverage = [], [], []
    for case in manifest['cases']:
        published = {arm: (args.results/case['case_id']/arm/'run.json').is_file() for arm in ('R', 'B', 'P', 'D')}
        publication_coverage.append(dict(case_id=case['case_id'], published=published))
        if not (args.results/case['case_id']/'RESULT.json').is_file():
            unavailable.append(case['case_id'])
            continue
        measurements.append(case_metrics(case, args.dataset, args.results))
    sources = {row['source_sha256'] for row in measurements}
    assert len(sources) == 1, 'Mixed or missing numerical source identities'
    qualification = False
    if args.qualification is not None:
        proof = json.loads(args.qualification.read_text())
        qualification = proof.get('status') == 'passed' and proof['source']['source_sha256'] in sources
    performance = None if args.performance is None else json.loads(args.performance.read_text())
    if performance is not None:
        assert performance['source_sha256'] in sources
    report = summarize(measurements, len(manifest['cases']), correctness_passed=qualification, performance=performance)
    report.update(unavailable=unavailable, measurements=measurements, publication_coverage=publication_coverage,
                  discovery_only=manifest['stage'] != 'confirmation',
                  manifest_sha256=sha(args.dataset/'EVALUATION_MANIFEST.json'))
    if report['discovery_only']:
        report['adoption_supported'] = False
    args.outdir.mkdir(parents=True, exist_ok=False)
    (args.outdir/'REPORT.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
