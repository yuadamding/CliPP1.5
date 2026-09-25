"""Saved-data-only input audit and truth-assisted diagnostics; never fits a model.

Run with the recorded ml1 interpreter. Output paths must be new or empty. The
fixed 193-case performance export defines the population, independent of live
campaign progress. Truth is used exclusively for retrospective diagnostics.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score


BASE = Path('/storage/CliPP2')
PREP = BASE / 'CliPP1.5/results/cnfirst-pool14-20260925-v1'
EXPORT = BASE / 'CNfirst4K_CliPP15_performance_20260925T181236Z/CLIPP15_RESULTS.json'
COHORT = BASE / 'CliPPSim4K_CNfirst_20260924'
PYCLONE = BASE / 'PyCloneVI_runs/CliPPSim4K_CNfirst_20260924_binomial_c20_r1000_cpu28'
PHYLOGIC = BASE / 'PhylogicNDT_runs/CliPPSim4K_CNfirst_20260924_ni1000_cpu28'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tab(path, **kwargs):
    return pd.read_csv(path, sep='\t', float_precision='round_trip', **kwargs)


def write_json(path, obj):
    with Path(path).open('x') as handle:
        json.dump(obj, handle, indent=2, allow_nan=False)
        handle.write('\n')


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'summary.json').exists(), 'Existing evidence is immutable'
    cases = {c['case_id']: c for c in read(PREP / 'CASES.json')}
    pv = {c['case_id']: c for c in read(PYCLONE / 'cases.json')}
    ph = {c['case_id']: c for c in read(PHYLOGIC / 'cases.json')}
    records = [g for g in read(EXPORT)['results']
               if g['terminal']['status'].startswith('validated_')]
    assert len(records) == 193
    checks = dict(cases=0, mutations=0, normal_cn_mismatch=0, purity_mismatch=0,
                  counts_mismatch=0, cn_mismatch=0, truth_mismatch=0,
                  pyclone_mismatch=0, phylogic_mismatch=0,
                  truth_outside_support=0, truth_outside_upper=0)
    max_purity_difference = 0.
    bindings, case_rows, mutation_rows, edge_rows = [], [], [], []
    pearson_z = []
    for saved in records:
        name = saved['case']['case_id']
        c = cases[name]
        raw = COHORT / name
        inp = PREP / 'inputs-v2' / c['input_filename']
        truth_path = PREP / 'truth-v2' / (name + '.tsv')
        expected = {str(inp): c['input_sha256'], str(truth_path): c['truth_sha256'],
                    pv[name]['input_file']: pv[name]['input_sha256'],
                    ph[name]['maf_file']: ph[name]['maf_sha256'],
                    ph[name]['sif_file']: ph[name]['sif_sha256']}
        expected.update(c['source_files'])
        for path, digest in expected.items():
            assert sha(path) == digest, path
        bindings.append(dict(case_id=name, files=expected))
        a = tab(inp, comment='#').set_index('position').sort_index()
        r = tab(raw / 'snv.txt').set_index('position').sort_index()
        t = tab(raw / 'truth.txt').set_index('position').sort_index()
        cn = tab(raw / 'cna.txt').set_index('start_position').sort_index()
        purity = float((raw / 'purity.txt').read_text())
        p = tab(pv[name]['input_file'])
        p['position'] = p.mutation_id.str.split(':').str[-1].astype(int)
        p = p.set_index('position').sort_index()
        h = tab(ph[name]['maf_file']).set_index('Start_position').sort_index()
        sif_purity = float(tab(ph[name]['sif_file']).purity.iloc[0])
        nt = tab(truth_path)
        nt['position'] = nt.mutation_id.str.split(':').str[-1].astype(int)
        nt = nt.set_index('position').sort_index()
        assert list(a.index) == list(r.index) == list(t.index) == list(p.index) == list(h.index) == list(nt.index)
        assert list(cn.index + 1) == list(a.index)
        assert (a.chromosome == 1).all() and (r.chromosome_index == 1).all()
        checks['normal_cn_mismatch'] += int((a.normal_cn != 2).sum())
        checks['purity_mismatch'] += int((np.abs(a.purity - purity) > 1e-12).sum())
        max_purity_difference = max(max_purity_difference, float(np.abs(a.purity - purity).max()))
        checks['counts_mismatch'] += int(((a.alt_count != r.alt_count) | (a.ref_count != r.ref_count)).sum())
        checks['cn_mismatch'] += int(((a.allele_a_cn.to_numpy() != cn.major_cn.to_numpy()) |
                                     (a.allele_b_cn.to_numpy() != cn.minor_cn.to_numpy())).sum())
        checks['truth_mismatch'] += int(((nt.true_ccf != t.ccf) | (nt.true_cluster != t.cluster_id) |
                                        (nt.true_multiplicity != t.multiplicity)).sum())
        checks['pyclone_mismatch'] += int(((a.alt_count != p.alt_counts) | (a.ref_count != p.ref_counts) |
                                          (a.normal_cn != p.normal_cn) | (a.allele_a_cn != p.major_cn) |
                                          (a.allele_b_cn != p.minor_cn) | (np.abs(a.purity - p.tumour_content) > 1e-12)).sum())
        # Phylogic calc_ccf documents a1=minor, a2=major.
        checks['phylogic_mismatch'] += int(((a.alt_count != h.t_alt_count) | (a.ref_count != h.t_ref_count) |
                                           (a.allele_a_cn != h.local_cn_a2) | (a.allele_b_cn != h.local_cn_a1) |
                                           (np.abs(a.purity - sif_purity) > 1e-12)).sum())
        major = a.allele_a_cn.to_numpy()
        minor = a.allele_b_cn.to_numpy()
        true_phi = t.ccf.to_numpy()
        true_mult = t.multiplicity.to_numpy()
        true_label = t.cluster_id.to_numpy()
        checks['truth_outside_support'] += int(((true_mult < 1) | (true_mult > major) | (major > 4)).sum())
        slope = purity / (2 * (1 - purity) + purity * (major + minor))
        upper = np.minimum(1., (1 - 1e-8) / (slope * major))
        checks['truth_outside_upper'] += int((true_phi > upper).sum())
        alt, ref = a.alt_count.to_numpy(), a.ref_count.to_numpy()
        centers, counts = np.unique(true_phi, return_counts=True)
        yt = np.searchsorted(centers, true_phi)
        multiplicities = np.arange(1, 5)
        valid = multiplicities[None, :] <= major[:, None]
        probability = np.clip(slope[:, None, None] * centers[None, :, None] * multiplicities[None, None, :], 1e-12, 1 - 1e-12)
        log_joint = alt[:, None, None] * np.log(probability) + ref[:, None, None] * np.log1p(-probability) - np.log(major[:, None, None])
        log_joint = np.where(valid[:, None, :], log_joint, -np.inf)
        likelihood = logsumexp(log_joint, axis=2)
        equal_assignment = likelihood.argmax(axis=1)
        frequency_assignment = (likelihood + np.log(counts / counts.sum())[None, :]).argmax(axis=1)
        cna = (major != 1) | (minor != 1)
        true_p = slope * true_phi * true_mult
        z = (alt - (alt + ref) * true_p) / np.sqrt((alt + ref) * true_p * (1 - true_p))
        pearson_z.extend(z.tolist())
        output = tab(io.StringIO(saved['tables']['mutation_clusters.tsv'])).set_index('mutation_id').loc[a.mutation_id]
        assert len(output) == len(a) and output.status.eq('retained').all()
        pilot = output.pilot_ccf.to_numpy()
        aliases = true_phi[:, None] * true_mult[:, None] / multiplicities[None, :]
        aliases_valid = valid & (aliases > 0) & (aliases <= 1)
        alias_distances = np.where(aliases_valid, np.abs(pilot[:, None] - aliases), np.inf)
        truth_distance = np.abs(pilot - true_phi)
        alt_distances = np.where(multiplicities[None, :] != true_mult[:, None], alias_distances, np.inf)
        closest_other = np.argmin(alt_distances, axis=1)
        other_distance = alt_distances[np.arange(len(a)), closest_other]
        alias_closer = other_distance < truth_distance - 1e-12
        lower_alias_closer = alias_closer & (aliases[np.arange(len(a)), closest_other] < true_phi)
        higher_alias_closer = alias_closer & (aliases[np.arange(len(a)), closest_other] > true_phi)
        # Source-equivalent all-pairs inverse-gap weights reconstructed from the
        # full-precision exported pilot. This is diagnostic, not a hash-certified
        # replay of the actual device tensor.
        gaps = np.diff(np.sort(pilot))
        positive = gaps[gaps > 0]
        gap_floor = max(1e-8, .1 * np.median(positive)) if len(positive) else 1e-8
        weights = 1 / np.maximum(np.abs(pilot[:, None] - pilot), gap_floor)
        np.fill_diagonal(weights, 0.)
        weights /= weights.sum() / (len(a) * (len(a) - 1))
        same = true_label[:, None] == true_label
        np.fill_diagonal(same, False)
        different = true_label[:, None] != true_label
        same_mean = (weights * same).sum(axis=1) / same.sum(axis=1)
        wrong_mean = np.divide((weights * different).sum(axis=1), different.sum(axis=1),
                               out=np.full(len(a), np.nan), where=different.sum(axis=1) > 0)
        low_m_cna = cna & (true_mult < major)
        diploid = ~cna
        for label, pair in {
            'same_truth_diploid_diploid': same & diploid[:, None] & diploid[None, :],
            'same_truth_low_m_cna_diploid': same & low_m_cna[:, None] & diploid[None, :],
            'different_truth_low_m_cna_diploid': different & low_m_cna[:, None] & diploid[None, :],
            'same_truth_cna_cna': same & cna[:, None] & cna[None, :],
            'different_truth_cna_cna': different & cna[:, None] & cna[None, :],
        }.items():
            values = weights[pair]
            edge_rows.append(dict(case_id=name, pair_type=label, pairs=len(values),
                                  weight_sum=float(values.sum()), weight_mean=float(values.mean()) if len(values) else None,
                                  weight_median=float(np.median(values)) if len(values) else None,
                                  gap_floor=float(gap_floor)))
        for i in range(len(a)):
            mutation_rows.append(dict(case_id=name, major_cn=int(major[i]), minor_cn=int(minor[i]), true_m=int(true_mult[i]),
                true_clonal=bool(true_phi[i] == 1), pilot_abs_error=float(truth_distance[i]),
                pilot_error_gt_0_1=bool(truth_distance[i] > .1), pilot_bias=float(pilot[i] - true_phi[i]),
                alias_closer=bool(alias_closer[i]), lower_alias_closer=bool(lower_alias_closer[i]), higher_alias_closer=bool(higher_alias_closer[i]),
                same_truth_mean_weight=float(same_mean[i]), different_truth_mean_weight=float(wrong_mean[i]),
                wrong_mean_weight_gt_same=bool(wrong_mean[i] > same_mean[i]) if len(centers) > 1 else None,
                true_k=len(centers)))
        case_rows.append(dict(case_id=name, n=len(a), true_k=len(centers),
            oracle_equal_ari=adjusted_rand_score(yt, equal_assignment), oracle_frequency_ari=adjusted_rand_score(yt, frequency_assignment),
            oracle_equal_error=float(np.mean(yt != equal_assignment)), oracle_frequency_error=float(np.mean(yt != frequency_assignment)),
            diploid_errors=int(((yt != frequency_assignment) & ~cna).sum()), diploid_n=int((~cna).sum()),
            cna_errors=int(((yt != frequency_assignment) & cna).sum()), cna_n=int(cna.sum()),
            pilot_mae=float(truth_distance.mean()), pilot_bias=float(np.mean(pilot - true_phi)),
            pilot_alias_closer=int(alias_closer.sum()), low_m_cna_n=int(low_m_cna.sum()),
            low_m_cna_alias_closer=int((low_m_cna & alias_closer).sum())))
        checks['cases'] += 1
        checks['mutations'] += len(a)
    failures = {key: value for key, value in checks.items() if key not in ('cases', 'mutations') and value}
    assert not failures, failures
    case_df, mutations, edges = pd.DataFrame(case_rows), pd.DataFrame(mutation_rows), pd.DataFrame(edge_rows)
    case_df.to_csv(out / 'per_case.tsv', sep='\t', index=False)
    grouped = mutations.groupby(['major_cn', 'true_m']).agg(
        mutations=('case_id', 'size'), pilot_mae=('pilot_abs_error', 'mean'), pilot_bias=('pilot_bias', 'mean'),
        pilot_error_gt_0_1=('pilot_error_gt_0_1', 'mean'), alias_closer_fraction=('alias_closer', 'mean'),
        lower_alias_closer_fraction=('lower_alias_closer', 'mean'), higher_alias_closer_fraction=('higher_alias_closer', 'mean'),
        same_truth_mean_weight=('same_truth_mean_weight', 'mean'), different_truth_mean_weight=('different_truth_mean_weight', 'mean'),
        wrong_mean_weight_gt_same_fraction=('wrong_mean_weight_gt_same', 'mean'))
    grouped.to_csv(out / 'pilot_by_major_multiplicity.tsv', sep='\t')
    edges.to_csv(out / 'graph_edges_per_case.tsv', sep='\t', index=False)
    edge_summary = edges.groupby('pair_type')[['pairs', 'weight_sum']].sum()
    edge_summary['pooled_mean_weight'] = edge_summary.weight_sum / edge_summary.pairs
    edge_summary.to_csv(out / 'graph_edges_summary.tsv', sep='\t')
    write_json(out / 'input_bindings.json', bindings)
    sums = case_df[['diploid_errors', 'diploid_n', 'cna_errors', 'cna_n']].sum().to_dict()
    source_files = [EXPORT, PREP / 'CASES.json', PYCLONE / 'cases.json', PHYLOGIC / 'cases.json',
                    COHORT / 'generation_manifest.json',
                    PREP / 'source/src/clipp1d/model.py', PREP / 'source/src/clipp1d/cuda/graph.py']
    hashes = {str(path): sha(path) for path in source_files if path.exists()}
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), checks=checks,
        max_purity_absolute_difference=max_purity_difference,
        binomial_residual_mean=float(np.mean(pearson_z)), binomial_residual_mean_square=float(np.mean(np.square(pearson_z))),
        oracle_equal_mean_ari=float(case_df.oracle_equal_ari.mean()), oracle_frequency_mean_ari=float(case_df.oracle_frequency_ari.mean()),
        oracle_equal_multi_mean_ari=float(case_df.loc[case_df.true_k > 1, 'oracle_equal_ari'].mean()),
        oracle_frequency_multi_mean_ari=float(case_df.loc[case_df.true_k > 1, 'oracle_frequency_ari'].mean()),
        classification_counts={key: int(value) for key, value in sums.items()},
        pilot=dict(mean_absolute_error=float(mutations.pilot_abs_error.mean()),
                   alias_closer_fraction=float(mutations.alias_closer.mean()),
                   lower_alias_closer_fraction=float(mutations.lower_alias_closer.mean()),
                   higher_alias_closer_fraction=float(mutations.higher_alias_closer.mean())),
        source_hashes=hashes, script_sha256=sha(__file__),
        definitions=dict(population='All 193 validated cases in the fixed export; all 41,712 retained mutations.',
            oracle='True centers are fixed. Assignment uses exact uniform-multiplicity marginal binomial log likelihood, either equal cluster probabilities or known true cluster frequencies. No optimization/refit. Not deployable performance and not an ARI upper bound.',
            alias='For observed truth CCF phi and multiplicity m, feasible alternate CCF phi*m/q for q in 1..major CN. Alias closer means a different q beats distance to truth by >1e-12.',
            graph='Source-equivalent inverse-pilot-gap weights, adjacent-positive-gap median floor and global mean-one normalization. Reconstructed from full-precision saved pilot; not a device-tensor-certified replay.',
            graph_pairs='Ordered focal/neighbor pairs; low-m CNA means CN not1/1 and true_m<major. Same/different refer to truth cluster membership.',
            wrong_weight='Per mutation average weight to wrong truth-cluster mutations exceeds average weight to own truth-cluster neighbors. K1 excluded.',
            purity='Numerical mismatch threshold 1e-12; serialization maximum separately reported.',
            cna='Any CN except1/1, including balanced amplification.'))
    write_json(out / 'summary.json', summary)
    write_json(out / 'ARTIFACTS.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'evidence')
    main(parser.parse_args().output)
