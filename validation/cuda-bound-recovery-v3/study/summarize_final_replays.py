"""Verify final literal-QP GPU replay receipts and independently audit saved arrays.

Read-only evidence arithmetic, never a QP/likelihood fit. Run beside the archived
reconcile_replays.py with --study-root pointing at the study or published archive.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reconcile_replays import (
    Attempt, canonical, certificate, replay_summary, require, sha, write_exact,
)


FINAL = '8627daf96b3b72be689a756445dfb60a7798a03d8bdd8d896f837eb1f11184c2'
DRIVER = '8bc12ec62bed6116409fa4dcc1bb7ef31fc28e87342a2c0f70e43db759d783c1'
GROUPS = (
    ('original_below64', 'capture-below-a', 'capture-below_one-64.json',
     '4958d20fdaf3f7d047ccd14a75dfb4a5ffadebd2f94c1ee8514a33fecda1b0af',
     'replay-final-below-m', 'replay-original-below64.json',
     '0588827851b6fc700a9ecf9e8e0f78d1fee94817ecd7423b218d40ee6fdaca04', 2),
    ('later_below64', 'capture-current-below-k', 'capture-current-below64.json',
     '0c63526d7e324f3d00125208165237df40800f49e2f919ef06f78711e752a4b6',
     'replay-final-below-m', 'replay-later-below64.json',
     '1d97d3342e3050c8f8b297496e8987f0fa855c2acca80fb9720d830cb363c025', 1),
    ('original_mixed256', 'capture-mixed-b', 'capture-mixed_support-256.json',
     'd457d0be06291349cefcce142aff8b8beeeb4f1cb88ab77e4da0acd69347cd26',
     'replay-final-mixed-n', 'replay-original-mixed256.json',
     'a5f6efde690612582fc4147fde878710483e66381e91ab0f31e9a7295813562e', 11),
)
PARAMETERS = ('h', 'target', 'lower', 'upper', 'caps', 'start', 'dual')


def audit_arrays(x, q, p, policy):
    h, target, lower, upper, caps = (p[k] for k in PARAMETERS[:5])
    free = lower < upper
    safe = np.where(free, target, x)
    a = -q.sum(axis=1)
    g = np.where(free, h*(x-safe), 0.)
    unboxed = safe-a/h
    z = np.clip(unboxed, lower, upper)
    delta = np.where(free, x-z, 0.)
    normal = np.where(unboxed < lower, h*(lower-safe)+a,
                      np.where(unboxed > upper, h*(upper-safe)+a, 0.))
    node_normal = normal*delta
    d = x[None, :]-x[:, None]
    edges = caps*np.abs(d)-q*d
    eps = np.finfo(np.float64).eps
    margin_n = 128*eps*(1+np.abs(h*(z-safe))+np.abs(a))*np.abs(delta)
    margin_e = 128*eps*(1+caps)*np.abs(d)
    gap = float(np.sum(.5*h*delta**2+np.maximum(node_normal, 0.)) + .5*np.maximum(edges, 0.).sum())
    reference = np.clip(safe, lower, upper)
    displacement = np.where(free, x-reference, 0.)
    energy = .5*h*displacement**2+h*(reference-safe)*displacement
    scale = float(np.maximum(energy, 0.).sum()+.5*np.where(free[:, None] | free[None, :], caps*np.abs(d), 0.).sum())
    r = g+a
    r = np.where(x == lower, np.minimum(r, 0.), r)
    r = np.where(x == upper, np.maximum(r, 0.), r)
    r = np.where(free, r, 0.)
    node = np.max(np.abs(r)/(1+np.abs(g)+np.abs(a)))
    dual = np.max(np.abs(q-np.clip(q+d, -caps, caps))/(1+caps))
    require(np.isfinite(x).all() and np.isfinite(q).all()
            and (x >= lower).all() and (x <= upper).all()
            and (np.abs(q) <= caps).all() and np.array_equal(q, -q.T)
            and (node_normal >= -margin_n).all() and (edges >= -margin_e).all(),
            'Independent saved-array feasibility/sign check failed')
    return certificate(dict(gap=gap, scale=scale, kkt=float(max(node, dual))), policy)


def shifted_objective(x, p):
    h, target, lower, upper, caps = (p[k] for k in PARAMETERS[:5])
    safe = np.where(lower < upper, target, lower)
    reference = np.clip(safe, lower, upper)
    delta = x-reference
    nodes = .5*h*delta**2+h*(reference-safe)*delta
    edges = .5*caps*(np.abs(x[None, :]-x[:, None])
                     - np.abs(reference[None, :]-reference[:, None]))
    value = float(nodes.sum()+edges.sum())
    margin = float(32*(len(x)+2)*np.finfo(np.float64).eps
                   * (1+np.abs(nodes).sum()+np.abs(edges).sum()))
    require(np.isfinite(value) and np.isfinite(margin), 'Nonfinite independent objective')
    return value, margin


def bound_driver(root, archive, attempt, digest):
    prepared = json.loads((attempt.parent/'PREPARED.json').read_text())
    candidates = [attempt.parent/'sealed'/prepared['run_id']/'source/benchmarks/replay_failed_qp.py',
                  archive/'reproduction/helpers'/digest/'replay_failed_qp.py']
    found = [path for path in candidates if path.is_file()]
    require(found, 'Missing exact replay driver bytes')
    for path in found:
        require(not path.is_symlink() and sha(path.read_bytes()) == digest, 'Replay driver bytes changed')
    # This source was independently reviewed: run() loads every literal problem
    # tensor, calls solve_qp with the exact saved start/dual, snapshots all inputs,
    # and rejects any changed tensor both after solve and after diagnostics.
    require(digest == DRIVER, 'Unreviewed replay driver')
    return dict(sha256=digest, identity_scope='Reviewed immutable driver passes the saved h/target/bounds/caps/start/dual directly and checks all tensor snapshots after execution')


def summarize(rows):
    return dict(total=len(rows), qualified=sum(row['final']['qualified'] for row in rows),
                admm_iterations_sum=sum(row['final']['admm_iterations'] for row in rows),
                admm_iterations_range=[min(row['final']['admm_iterations'] for row in rows),
                                       max(row['final']['admm_iterations'] for row in rows)],
                polish_iterations_sum=sum(row['final']['polish_iterations'] for row in rows),
                polish_iterations_range=[min(row['final']['polish_iterations'] for row in rows),
                                         max(row['final']['polish_iterations'] for row in rows)],
                recorded_solve_seconds_sum=sum(row['final']['seconds'] for row in rows),
                all_original_objectives_nonincreasing=all(row['objective']['nonincreasing'] for row in rows),
                maximum_compiled_gap_to_allowance=max(row['final']['compiled_certificate']['gap_to_allowance'] for row in rows),
                maximum_eager_gap_to_allowance=max(row['final']['eager_certificate']['gap_to_allowance'] for row in rows),
                maximum_compiled_kkt=max(row['final']['compiled_certificate']['kkt'] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    archive = args.study_root.resolve()
    root = archive/'attempts' if (archive/'attempts').is_dir() else archive
    rows, identities, policies, helpers = [], [], [], []
    for family, capture_name, capture_file, capture_sha, replay_name, replay_file, replay_sha, count in GROUPS:
        captured = Attempt(root, capture_name, capture_file)
        replay = Attempt(root, replay_name, replay_file)
        cap, receipt = captured.receipt, replay.receipt
        require(captured.receipt_sha256 == capture_sha and replay.receipt_sha256 == replay_sha,
                'Exact approved replay/capture receipt changed')
        require(receipt['source']['source_sha256'] == FINAL and receipt['numerical_execution'] == 'CUDA float64',
                'Final source or execution device differs')
        require(receipt['capture']['sha256'] == capture_sha
                and receipt['capture']['baseline_source_sha256'] == cap['source']['source_sha256'],
                'Replay uses a different capture source')
        policy = cap['policy']
        require(policy == receipt['policy'] and policy['inner_atol'] == 1e-10
                and policy['inner_rtol'] == 1e-11 and policy['inner_kkt_tol'] == 1e-7
                and policy['inner_max_iterations'] == 20000, 'Original policy/gates differ')
        require(cap['diagnostic_capture_qualified'] and not cap['scientific_fit_qualified']
                and len(cap['captures']) == count == receipt['captured_failed_qps']
                == receipt['resolved_qps'] == len(receipt['replays'])
                and receipt['unresolved_qps'] == 0 and receipt['all_replays_qualified'], 'Replay coverage differs')
        require(all(not item['eager_fallback'] and item['fullgraph']
                    for item in receipt['compilation_statistics'].values()), 'Compiler fallback occurred')
        helper = bound_driver(root, archive, replay, receipt['script_sha256'])
        helpers.append(helper)
        identities.extend([captured.identity(), replay.identity()])
        policies.append(policy)
        replay_rows = {item['capture_record_sha256']: item for item in receipt['replays']}
        require(len(replay_rows) == count and sorted(replay_rows) == sorted(item['sha256'] for item in cap['captures']),
                'Missing or duplicate captured QP')
        for index, entry in enumerate(cap['captures']):
            record = captured.load('results/'+entry['record'], entry['sha256'])
            require(record['context'] == entry['context'] and record['context']['policy'] == policy
                    and record['context']['source_sha256'] == cap['source']['source_sha256']
                    and record['context']['input_sha256'] == cap['input_sha256'], 'Literal context identity differs')
            require(set(record['problem']) == set(PARAMETERS), 'Missing literal problem/initialization field')
            problem = {key: None if value is None else np.asarray(captured.tensor(value), dtype=np.float64)
                       for key, value in record['problem'].items()}
            hashes = {key: None if value is None else value['tensor_sha256'] for key, value in record['problem'].items()}
            item = replay_rows[entry['sha256']]
            require(item['capture_index'] == index, 'Replay captured-QP order differs')
            final = replay_summary(replay, item, entry, captured, record, policy, True)
            initial_stats = certificate({key: captured.tensor(record['returned'][key]) for key in ('gap', 'scale', 'kkt')}, policy)
            require(not initial_stats['qualified'] and not record['returned']['qualified'], 'Original failure was qualified')
            x, q = (np.asarray(replay.tensor(item['result'][key]), dtype=np.float64) for key in ('x', 'q'))
            independent = audit_arrays(x, q, problem, policy)
            require(independent['qualified'], 'Independent literal-array certificate failed')
            old_x = np.asarray(captured.tensor(record['returned']['x']), dtype=np.float64)
            old_objective, old_margin = shifted_objective(old_x, problem)
            new_objective, new_margin = shifted_objective(x, problem)
            old_recorded = item['original_baseline']['original_shifted_objective']
            new_recorded = item['original_shifted_objective']
            require(abs(old_objective-old_recorded) <= old_margin
                    and abs(new_objective-new_recorded) <= new_margin, 'Independent objective reconstruction differs')
            require(new_recorded <= old_recorded, 'Qualified result worsened the captured objective')
            rows.append(dict(family=family, capture_index=index, capture_record_sha256=entry['sha256'],
                             capture_receipt_sha256=capture_sha, replay_receipt_sha256=replay_sha,
                             context=record['context'], context_sha256=sha(canonical(record['context'])),
                             problem_and_original_initialization_tensor_sha256=hashes,
                             original_failed_certificate=initial_stats,
                             original_admm_iterations=record['returned']['iterations'],
                             original_polish_iterations=record['returned']['polish_iterations'],
                             final=final, independent_numpy_certificate=independent,
                             objective=dict(captured=old_recorded, final=new_recorded,
                                            change=new_recorded-old_recorded, nonincreasing=True,
                                            independent_captured=old_objective, independent_final=new_objective,
                                            independent_captured_reduction_error_margin=old_margin,
                                            independent_final_reduction_error_margin=new_margin)))
    require(len(rows) == 14 and len({row['capture_record_sha256'] for row in rows}) == 14,
            'Final capture inventory is not exactly 14 unique QPs')
    require(all(policy == policies[0] for policy in policies) and all(helper == helpers[0] for helper in helpers),
            'Policy or reviewed replay driver differs between receipts')
    result = dict(schema='clipp1d.final_literal_qp_replay_status.v1', status='passed',
                  scope='Exactly 14 literal saved QPs replayed from original starts/duals on allocated CUDA; no full-path, refit, publication, or cohort claim.',
                  source_sha256=FINAL, summary=summarize(rows),
                  families={family: summarize([row for row in rows if row['family'] == family]) for family, *_ in GROUPS},
                  policy=policies[0], inputs=identities, reviewed_replay_driver=helpers[0],
                  identity_checks=dict(exact_receipt_artifact_and_tensor_hashes=True,
                                       exact_problem_original_start_and_dual_bound_by_reviewed_driver=True,
                                       original_terminal_certificate_reconstructed=True,
                                       eager_and_compiled_original_gates_pass=True,
                                       independent_saved_array_feasibility_and_certificate_pass=True,
                                       unchanged_gates_and_per_qp_admm_budget=True),
                  timing_scope='Receipt solve times may include lazy compilation; this is not a repeated latency benchmark.',
                  script_sha256=sha(Path(__file__).read_bytes()),
                  support_script_sha256=sha(Path(__file__).with_name('reconcile_replays.py').read_bytes()),
                  numpy_version=np.__version__, rows=rows)
    out = args.out or archive/'FINAL_REPLAY_STATUS.json'
    write_exact(out, (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n').encode())
    print(json.dumps(dict(output=str(out), sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                          summary=result['summary'], families=result['families']), sort_keys=True))


if __name__ == '__main__':
    main()
