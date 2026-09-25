import json

import pytest

from benchmarks.run_complete_graph_cpu import borrowed_cpus


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def parent(tmp_path):
    r = tmp_path / "receipts"
    r.mkdir()
    write(
        r / "controller-progress.json",
        dict(
            phase="halted_draining",
            halt_reason="requested",
            active={"1": dict(cpu=3), "2": dict(cpu=7)},
        ),
    )
    write(r / "1.accepted.json", {})
    write(r / "2.accepted.json", {})
    write(r / "controller-terminal.json", {})
    return dict(draining_parent=str(tmp_path), draining_parent_accepted=["1", "2"])


def test_running_parent_workers_keep_their_cpu_slots(parent):
    assert borrowed_cpus(parent) == {3, 7}
    assert not ({0, 1, 2, 3, 7} - borrowed_cpus(parent)) & {3, 7}


def test_handoff_refuses_parent_that_resumed_admissions(parent):
    from pathlib import Path

    r = Path(parent["draining_parent"]) / "receipts"
    write(r / "3.accepted.json", {})
    with pytest.raises(AssertionError, match="admitted new work"):
        borrowed_cpus(parent)


def test_initial_pool_has_no_borrowed_slots():
    assert borrowed_cpus({}) == set()


def test_deferred_case_waits_and_retries_only_after_parent_failure(parent):
    from pathlib import Path
    from benchmarks.run_complete_graph_cpu import deferred_parent_outcome

    assert deferred_parent_outcome(parent, '1') is None
    r = Path(parent['draining_parent'])/'receipts'
    write(r/'1.reconciled.json', dict(outcome=dict(status='resource_timeout')))
    assert deferred_parent_outcome(parent, '1')['action'] == 'retry'


def test_deferred_case_does_not_bypass_parent_integrity_failure(parent):
    from pathlib import Path
    from benchmarks.run_complete_graph_cpu import deferred_parent_outcome

    r = Path(parent['draining_parent'])/'receipts'
    write(r/'1.reconciled.json', dict(outcome=dict(status='output_validation_failure')))
    with pytest.raises(AssertionError, match='integrity failure'):
        deferred_parent_outcome(parent, '1')


def test_deferred_case_preserves_exhausted_retry_budget(parent):
    from pathlib import Path
    from benchmarks.run_complete_graph_cpu import deferred_parent_outcome

    r = Path(parent['draining_parent'])/'receipts'
    write(r/'1.reconciled.json', dict(outcome=dict(status='resource_timeout')))
    parent['deferred_parent_retry_keys'] = []
    assert deferred_parent_outcome(parent, '1')['action'] == 'retain_failure'


def test_deferred_success_reuses_only_hash_validated_outputs(parent):
    from pathlib import Path
    from benchmarks.run_complete_graph_cpu import deferred_parent_outcome, digest

    root = Path(parent['draining_parent'])
    task = root/'results/1'
    (task/'output').mkdir(parents=True)
    (task/'output/result.tsv').write_text('result\n')
    write(task/'validated.json', dict(output_sha256={'result.tsv': digest(task/'output/result.tsv')}))
    write(root/'receipts/1.reconciled.json', dict(outcome=dict(status='validated_complete', validated_sha256=digest(task/'validated.json'))))
    assert deferred_parent_outcome(parent, '1')['action'] == 'reuse'
    (task/'output/result.tsv').write_text('changed\n')
    with pytest.raises(AssertionError):
        deferred_parent_outcome(parent, '1')
