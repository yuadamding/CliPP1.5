"""Explicit CPU adapter contracts; these tests do not qualify GPU execution."""
import json
from pathlib import Path

import pytest
import torch

from benchmarks import fit_complete_graph_cpu as cpu


@pytest.fixture
def input_file(tmp_path):
    path = tmp_path/'preserved-name.tsv'
    path.write_text('mutation_id\tsample_id\talt_count\tref_count\tcount_observed\tpurity\tnormal_cn\tsegment_id\tcn_state_id\tcn_state_fraction\tallele_a_cn\tallele_b_cn\n'
                    'a\tregion1\t20\t80\t1\t1\t2\ts1\tc1\t1\t1\t1\n'
                    'b\tregion1\t48\t52\t1\t1\t2\ts2\tc1\t1\t1\t1\n')
    return path


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def test_complete_cpu_path_keeps_identity_gates_and_explicit_backend(input_file, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('CPU fit requested CUDA execution or device information')
    for name in ('synchronize', 'get_device_properties', 'mem_get_info', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    source = cpu.source_provenance()['source_sha256']
    result = cpu.fit(input_file, tmp_path/'output', expected_source_sha256=source,
                     memory_budget_bytes=1024**3)
    record = json.loads((tmp_path/'output/run.json').read_text())
    assert result.search_status == 'complete'
    assert len(record['search']) >= 26
    assert result.provenance['backend'] == 'cpu'
    assert result.provenance['execution_scope'] == 'cpu_numerical_reference'
    assert not result.provenance['compiled_inference']
    assert not result.provenance['cpu_numeric_fallback']
    assert record['raw_diagnostics']['status'] in ('qualified', 'qualified_separable')
    assert result.candidate_provenance['raw_qualified'] and result.candidate_provenance['refit_qualified']
    assert record['provenance']['policy'] == cpu.asdict(cpu.CudaPolicy())
    assert 'preserved-name\tregion1' in (tmp_path/'output/mutation_clusters.tsv').read_text()
    assert set(record['table_sha256']) == {'mutation_clusters.tsv', 'cluster_centers.tsv', 'mutation_multiplicity.tsv'}


def test_source_mismatch_and_memory_admission_do_not_fit(input_file, tmp_path, monkeypatch):
    monkeypatch.setattr(cpu, 'fit_tensor_model', lambda *a: pytest.fail('Unadmitted fit'))
    with pytest.raises(ValueError, match='frozen source'):
        cpu.fit(input_file, tmp_path/'wrong-source', expected_source_sha256='wrong', memory_budget_bytes=1024**3)
    with pytest.raises(MemoryError, match='worker budget'):
        cpu.fit(input_file, tmp_path/'too-small', expected_source_sha256=cpu.source_provenance()['source_sha256'], memory_budget_bytes=1)
    for name in ('wrong-source', 'too-small'):
        receipt = json.loads((tmp_path/name/'run.json').read_text())
        assert receipt['status'] == 'failure' and receipt['provenance']['backend'] == 'cpu'
        assert not list((tmp_path/name).glob('*.tsv'))


def test_cpu_output_is_not_overwritten(input_file, tmp_path):
    out = tmp_path/'prior'
    out.mkdir()
    (out/'evidence.txt').write_text('keep')
    with pytest.raises(FileExistsError):
        cpu.fit(input_file, out, expected_source_sha256='unused', memory_budget_bytes=1)
    assert (out/'evidence.txt').read_text() == 'keep'
    link = tmp_path/'link'
    link.symlink_to(out)
    with pytest.raises(FileExistsError, match='symlink'):
        cpu.fit(input_file, link, expected_source_sha256='unused', memory_budget_bytes=1)


def test_invalid_budget_is_rejected_before_output_creation(input_file, tmp_path):
    out = tmp_path/'absent'
    with pytest.raises(ValueError, match='positive'):
        cpu.fit(input_file, out, expected_source_sha256='unused', memory_budget_bytes=0)
    assert not Path(out).exists()
