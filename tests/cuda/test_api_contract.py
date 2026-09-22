"""Failure/publication contracts that can be tested without a CUDA device."""
import json
import pytest
import torch
from clipp1d.cuda_api import fit, require_cuda
from clipp1d.api import source_provenance


def test_cpu_is_not_a_production_fallback():
    with pytest.raises(ValueError, match="CUDA"):
        require_cuda("cpu")


def test_requested_cuda_unavailable_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda:False)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        fit("unused.tsv", output)
    receipt = json.loads((output / "run.json").read_text())
    assert receipt['status'] == 'failure'
    assert receipt['search_status'] == 'not_completed'
    assert not list(output.glob('*.tsv'))


def test_no_output_overwrite(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / 'old.txt').write_text('evidence')
    with pytest.raises(FileExistsError):
        fit("unused.tsv", output)
    assert (output / 'old.txt').read_text() == 'evidence'


def test_output_symlink_rejected_before_failure_receipt(tmp_path, monkeypatch):
    target = tmp_path / 'untouched'
    target.mkdir()
    link = tmp_path / 'output-link'
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(FileExistsError, match='symlink'):
        fit('unused.tsv', link)
    assert list(target.iterdir()) == []


def test_provenance_includes_nested_cuda_sources():
    source = source_provenance()
    assert 'cuda/qp.py' in source['source_files']
    assert 'cuda/scalar.py' in source['source_files']
    assert len(source['source_sha256']) == 64


def test_successful_publication_from_tensor_reference(tmp_path):
    # Exercises serialization after a CPU numerical reference, not public CUDA fit.
    from types import SimpleNamespace
    from clipp1d.cuda_api import _export, _publish
    from clipp1d.cuda.model import TensorModel
    from clipp1d.cuda.kernels import Kernels
    from clipp1d.cuda.selection import fit_tensor_model
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        alt=torch.tensor([20.,48.],dtype=torch.float64)
        slope=torch.full((2,1),.5,dtype=torch.float64)
        model=TensorModel(('a','b'),alt,100-alt,slope,torch.zeros_like(slope),
                          torch.full_like(alt,1e-6),torch.ones_like(alt),1e-6,Kernels('cpu'))
        fitted=fit_tensor_model(model,lambda_values=[0.])
        result=_export(fitted,source_provenance())
        data=SimpleNamespace(tumor_id='tumor',sample_id='sample',
                             mutations=[SimpleNamespace(mutation_id='a',exclusion=None),
                                        SimpleNamespace(mutation_id='b',exclusion=None),
                                        SimpleNamespace(mutation_id='excluded',exclusion='ZERO_DEPTH')])
        _publish(result,data,tmp_path,fitted.records)
        receipt=json.loads((tmp_path/'run.json').read_text())
        assert receipt['status']=='success'
        assert len(receipt['table_sha256'])==3
        rows=(tmp_path/'mutation_clusters.tsv').read_text().splitlines()
        assert 'raw_ccf' in rows[0] and 'raw_reference_ccf' not in rows[0]
        assert len(rows[0].split('\t'))==len(rows[-1].split('\t'))
        assert receipt['provenance']['primary_estimator']=='selected_raw_complete_graph_penalized_candidate'
    finally:
        torch.set_num_threads(old)


@pytest.fixture
def reference_bundle():
    """Explicit CPU tensor reference; never advertised as CUDA qualification."""
    from types import SimpleNamespace
    from clipp1d.cuda_api import _export
    from clipp1d.cuda.model import TensorModel
    from clipp1d.cuda.kernels import Kernels
    from clipp1d.cuda.selection import fit_tensor_model
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        alt = torch.tensor([20., 48.], dtype=torch.float64)
        slope = torch.full((2, 1), .5, dtype=torch.float64)
        model = TensorModel(('a', 'b'), alt, 100 - alt, slope, torch.zeros_like(slope),
                            torch.full_like(alt, 1e-6), torch.ones_like(alt), 1e-6,
                            Kernels('cpu', compiled=False))
        fitted = fit_tensor_model(model, lambda_values=[0.])
        source = source_provenance()
        result = _export(fitted, source)
        data = SimpleNamespace(tumor_id='tumor', sample_id='sample',
                               mutations=[SimpleNamespace(mutation_id='a', exclusion=None),
                                          SimpleNamespace(mutation_id='b', exclusion=None)])
        yield fitted, source, result, data
    finally:
        torch.set_num_threads(previous)


def test_exported_numeric_arrays_are_immutable(reference_bundle):
    _, _, result, _ = reference_bundle
    for name in ('raw_phi', 'pilot_phi', 'refitted_phi', 'cluster_labels', 'cluster_centers',
                 'original_lower_bounds', 'original_upper_bounds', 'partition_labels',
                 'multiplicity_calls', 'refitted_multiplicity_calls'):
        value = getattr(result, name)
        with pytest.raises(ValueError):
            value.setflags(write=True)


def test_export_copies_metadata_and_rejects_later_modification(reference_bundle):
    from clipp1d.cuda_api import _validate_result
    from clipp1d.cuda.policy import QualificationError
    fitted, source, result, data = reference_bundle
    source['source_sha256'] = 'changed external caller object'
    fitted.raw.diagnostics['new_external_key'] = 'must not alter result'
    _validate_result(result, data, fitted.records)
    assert 'new_external_key' not in result.raw_diagnostics
    assert result.provenance['source_sha256'] != source['source_sha256']
    result.raw_diagnostics['box_feasible'] = False
    with pytest.raises(QualificationError, match='modified scientific provenance'):
        _validate_result(result, data, fitted.records)


def test_export_rejects_forged_uploaded_model_identity(reference_bundle):
    from clipp1d.cuda_api import _export
    from clipp1d.cuda.policy import QualificationError
    fitted, source, _, _ = reference_bundle
    source['model_sha256'] = '0' * 64
    with pytest.raises(QualificationError, match='model does not match'):
        _export(fitted, source)


def test_cpu_reference_cannot_claim_compiled_cuda(reference_bundle):
    from clipp1d.cuda_api import _export
    from clipp1d.cuda.policy import QualificationError
    fitted, source, result, _ = reference_bundle
    assert result.provenance['execution_scope'] == 'cpu_numerical_reference'
    assert result.provenance['compiled_inference'] is False
    source['backend'] = 'cuda'
    with pytest.raises(QualificationError, match='cannot inherit compiled CUDA'):
        _export(fitted, source)


@pytest.mark.parametrize('field', ['raw_objective', 'refit_score', 'refit_qualification', 'refit_lower_bound'])
def test_export_reaudits_device_candidate_arithmetic(reference_bundle, field):
    from clipp1d.cuda_api import _export
    from clipp1d.cuda.policy import QualificationError
    fitted, source, _, _ = reference_bundle
    if field == 'raw_objective':
        fitted.raw.objective += 1.
    elif field == 'refit_score':
        fitted.refit.score += 1.
    elif field == 'refit_qualification':
        fitted.refit.scalar.qualified[0] = False
    else:
        fitted.refit.scalar.lower_bound[0] -= 1.
    with pytest.raises(QualificationError):
        _export(fitted, source)


def test_model_data_writes_cannot_bypass_immutability_guard(reference_bundle):
    from clipp1d.cuda_api import _export
    fitted, source, _, _ = reference_bundle
    fitted.model.alt.data[0] += 1.
    with pytest.raises(ValueError, match='modified'):
        _export(fitted, source)


def test_publication_rejects_bound_or_multiplicity_replacement(reference_bundle):
    from dataclasses import replace
    from clipp1d.cuda_api import _validate_result
    from clipp1d.cuda.policy import QualificationError
    fitted, _, result, data = reference_bundle
    for field in ('original_lower_bounds', 'multiplicity_calls'):
        changed = getattr(result, field).copy()
        changed[0] = changed[0] + (1e-7 if field == 'original_lower_bounds' else 1)
        altered = replace(result, **{field: changed})
        with pytest.raises(QualificationError, match='state identity'):
            _validate_result(altered, data, fitted.records)


def test_selected_path_record_must_be_actual_score_winner(reference_bundle):
    from copy import deepcopy
    from clipp1d.cuda_api import _validate_result
    from clipp1d.cuda.policy import QualificationError
    fitted, _, result, data = reference_bundle
    records = deepcopy(fitted.records)
    better = deepcopy(records[0])
    better.update(lambda_value=.25, score=result.selection_score - 1.)
    records.append(better)
    with pytest.raises(QualificationError, match='does not minimize'):
        _validate_result(result, data, records)


def test_failed_staging_publishes_no_partial_tables(reference_bundle, tmp_path, monkeypatch):
    import clipp1d.cuda_api as api
    fitted, _, result, data = reference_bundle
    original_json = api._json
    def fail_receipt(path, value):
        if value.get('status') == 'success':
            raise OSError('simulated receipt write failure')
        return original_json(path, value)
    monkeypatch.setattr(api, '_json', fail_receipt)
    target = tmp_path / 'output'
    with pytest.raises(OSError, match='simulated'):
        api._publish(result, data, target, fitted.records)
    assert list(target.iterdir()) == []
    assert list(tmp_path.glob('.clipp1d-publish-*')) == []


def test_publication_race_does_not_overwrite_existing_evidence(reference_bundle, tmp_path, monkeypatch):
    import clipp1d.cuda_api as api
    fitted, _, result, data = reference_bundle
    target = tmp_path / 'output'
    original_replace = api.os.replace
    def concurrent_writer(source, destination):
        (target / 'other-run.txt').write_text('preserve this evidence')
        return original_replace(source, destination)
    monkeypatch.setattr(api.os, 'replace', concurrent_writer)
    with pytest.raises(OSError):
        api._publish(result, data, target, fitted.records)
    assert (target / 'other-run.txt').read_text() == 'preserve this evidence'
    assert not list(target.glob('*.tsv'))
    assert not (target / 'run.json').exists()
    assert list(tmp_path.glob('.clipp1d-publish-*')) == []


def test_public_qualification_exception_preserves_existing_catch_contract():
    from clipp1d import QualificationError, NumericalQualificationError
    error = QualificationError('unresolved', gap=1.)
    assert isinstance(error, NumericalQualificationError)
    assert isinstance(error, ValueError)
    assert error.diagnostics == {'gap': 1.}
