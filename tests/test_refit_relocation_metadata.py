"""Standard-library-only policy and coverage controls; no numerical imports."""
import ast
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest


class QualificationError(RuntimeError):
    pass


def load_definition(relative, name, globals_):
    path = Path(__file__).resolve().parents[1] / relative
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    scope = dict(globals_)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


Policy = load_definition("src/clipp1d/cuda/refinement.py", "PartitionSearchPolicy", {"dataclass": dataclass})
count = load_definition("src/clipp1d/cuda/refit_relocation.py", "neighborhood_size", {})
validate = load_definition("src/clipp1d/partition_output.py", "validate_relocation_summary",
                           {"QualificationError": QualificationError})
schema = load_definition("src/clipp1d/partition_output.py", "partition_estimate_schema", {})


def identity(k, score):
    return dict(n=5, k=k, score=score, score_lower=score, score_upper=score, refit_gap=0.,
                refit_qualified=True, membership_sha256=str(k)*64)


def fixture(max_candidates=15, max_scans=2):
    parent, child = identity(3, 100.), identity(2, 90.)
    first = dict(parent=parent, child=child, planned_candidates=10, evaluated_candidates=10,
        infeasible_candidates=0, unresolved_candidates=[], accepted=True, scan_complete=True,
        selected_endpoint_complete=False, status="accepted", remaining_candidates=0,
        scan_index=0, candidate_budget_before=max_candidates, candidate_budget_after=max_candidates-10)
    scans = [first]
    status, charged, complete = 'scan_budget_exhausted', 10, False
    if max_scans > 1:
        enough = max_candidates >= 15
        second = dict(parent=child, planned_candidates=5, evaluated_candidates=5 if enough else 0,
            infeasible_candidates=0, unresolved_candidates=[], accepted=False, scan_complete=enough,
            selected_endpoint_complete=enough, status='fixed_point' if enough else 'candidate_budget_exhausted',
            remaining_candidates=0 if enough else 5, scan_index=1,
            candidate_budget_before=max_candidates-10, candidate_budget_after=max_candidates-(15 if enough else 10))
        scans.append(second)
        status, charged, complete = second['status'], 15 if enough else 10, enough
    summary = dict(enabled=True, neighborhood="all_mutations_to_existing_groups",
        fixed_point_criterion="no_interval_certified_decrease_above_margin", status=status, scans=scans,
        max_scans=max_scans, max_candidates=max_candidates, charged_candidates=charged,
        remaining_candidate_budget=max_candidates-charged, historical_unresolved=0,
        selected_endpoint=child, selected_endpoint_complete=complete, global_partition_optimum_proven=False)
    policy = asdict(Policy(refit_relocations=True, refit_max_scans=max_scans,
                           refit_max_candidates=max_candidates))
    records = [dict(search_phase='refit_relocation', details=deepcopy(s)) for s in scans]
    records.append(dict(search_phase='refit_relocation', status='qualified',
        candidate_family='direct_partition', proposal=dict(ancestry=[dict(operation='refit_relocation',
            parent=parent, child=child, details=deepcopy(first))])))
    return summary, policy, child, records


def test_default_disabled_and_k_adaptive_candidate_count():
    assert Policy().refit_relocations is False
    assert count(5, 3) == 10 and count(5, 2) == 5 and count(5, 1) == 0
    validate(None, asdict(Policy()), identity(2, 90.), [])


def test_new_ancestry_schema_is_enabled_only_for_explicit_relocation():
    assert schema({}) == schema(asdict(Policy())) == 'clipp1d.partition_estimate.v2'
    assert schema(asdict(Policy(refit_relocations=True))) == 'clipp1d.partition_estimate.v3'


@pytest.mark.parametrize('kwargs', [dict(refit_relocations=1), dict(refit_max_scans=0),
    dict(refit_max_candidates=True), dict(refit_batch_candidates=-1)])
def test_invalid_refit_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        Policy(**kwargs)


@pytest.mark.parametrize('args', [(0, 0), (3, 4), (True, 1), (3, 1.)])
def test_invalid_neighborhood_size_rejected(args):
    with pytest.raises(ValueError):
        count(*args)


@pytest.mark.parametrize('budget,scans', [(15, 2), (14, 2), (10, 1)])
def test_valid_fixedpoint_candidate_budget_and_accepted_unscanned_ledgers(budget, scans):
    validate(*fixture(budget, scans))


@pytest.mark.parametrize('field,value', [('selected_endpoint_complete', True),
    ('status', 'fixed_point'), ('charged_candidates', 9), ('historical_unresolved', False),
    ('global_partition_optimum_proven', True)])
def test_accepted_child_cannot_inherit_parent_coverage(field, value):
    summary, policy, selected, records = fixture(10, 1)
    summary[field] = value
    with pytest.raises(QualificationError):
        validate(summary, policy, selected, records)


def test_global_budget_and_scan_record_hash_scope_cannot_drift():
    summary, policy, selected, records = fixture()
    summary['scans'][1]['parent'] = identity(3, 100.)
    records[1]['details'] = deepcopy(summary['scans'][1])
    with pytest.raises(QualificationError, match='parent continuity'):
        validate(summary, policy, selected, records)
    summary, policy, selected, records = fixture()
    summary['scans'][1]['evaluated_candidates'] = 4
    with pytest.raises(QualificationError, match='unbound scan records'):
        validate(summary, policy, selected, records)


def test_historical_unresolved_is_preserved_even_if_final_endpoint_complete():
    summary, policy, selected, records = fixture()
    summary['scans'][0]['unresolved_candidates'] = [dict(node=0)]
    records[0]['details'] = deepcopy(summary['scans'][0])
    records[-1]['proposal']['ancestry'][-1]['details'] = deepcopy(summary['scans'][0])
    with pytest.raises(QualificationError, match='final phase ledger'):
        validate(summary, policy, selected, records)
    summary['historical_unresolved'] = 1
    summary['status'] = 'unresolved'
    validate(summary, policy, selected, records)
    assert summary['selected_endpoint_complete']


def test_disabled_scope_rejects_claims():
    summary, _, selected, records = fixture()
    with pytest.raises(QualificationError, match='disabled phase'):
        validate(summary, asdict(Policy()), selected, records)


def test_rewriting_scan_and_summary_cannot_detach_true_accepted_ancestry():
    summary, policy, selected, records = fixture()
    summary['scans'][0]['parent'] = identity(3, 101.)
    records[0]['details'] = deepcopy(summary['scans'][0])
    with pytest.raises(QualificationError, match='ancestry binding'):
        validate(summary, policy, selected, records)


@pytest.mark.parametrize('operation', ['extra', 'missing'])
def test_no_extra_or_missing_direct_phase_record(operation):
    summary, policy, selected, records = fixture()
    if operation == 'extra':
        records.append(deepcopy(records[-1]))
    else:
        records.pop()
    with pytest.raises(QualificationError, match='direct record count'):
        validate(summary, policy, selected, records)


def test_multiple_accepted_scans_bind_the_entire_selected_ancestry():
    summary, policy, _, records = fixture()
    second = summary['scans'][1]
    second.update(accepted=True, status='accepted', selected_endpoint_complete=False,
                  child=identity(2, 80.))
    records[1]['details'] = deepcopy(second)
    last = deepcopy(records[-1])
    last['proposal']['ancestry'].append(dict(operation='refit_relocation', parent=second['parent'],
                                            child=second['child'], details=deepcopy(second)))
    records.append(last)
    summary.update(status='scan_budget_exhausted', selected_endpoint_complete=False,
                   selected_endpoint=second['child'])
    validate(summary, policy, second['child'], records)
    # Forge the first scan and its own direct record, leaving selected history true.
    summary['scans'][0]['parent'] = identity(3, 101.)
    records[0]['details'] = deepcopy(summary['scans'][0])
    records[2]['proposal']['ancestry'][-1]['parent'] = summary['scans'][0]['parent']
    records[2]['proposal']['ancestry'][-1]['details'] = deepcopy(summary['scans'][0])
    with pytest.raises(QualificationError, match='ancestry continuity'):
        validate(summary, policy, second['child'], records)
