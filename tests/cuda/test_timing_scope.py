"""Measurement boundaries include final device work and separate durable publication."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch


@pytest.fixture
def device_reference():
    from clipp1d.cuda.model import TensorModel
    from clipp1d.cuda.kernels import Kernels
    from clipp1d.cuda.selection import fit_tensor_model
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        alt = torch.tensor([20., 48.], dtype=torch.float64)
        slope = torch.full((2, 1), .5, dtype=torch.float64)
        model = TensorModel(('a', 'b'), alt, 100 - alt, slope, torch.zeros_like(slope),
                            torch.full_like(alt, 1e-6), torch.ones_like(alt), 1e-6, Kernels('cpu'))
        fitted = fit_tensor_model(model, lambda_values=[0.])
        data = SimpleNamespace(tumor_id='tumor', sample_id='sample', mutations=[
            SimpleNamespace(mutation_id=mid, exclusion=None) for mid in model.mutation_ids])
        yield fitted, data
    finally:
        torch.set_num_threads(previous)


def test_final_device_work_precedes_peak_and_compilation_snapshot(device_reference, monkeypatch):
    import clipp1d.cuda_api as api
    from clipp1d.api import source_provenance
    fitted, _ = device_reference
    clock = [0.]
    events = []
    validate, host, metrics = api._validate_device_fit, api._host, api._device_metrics

    def final_validation(d):
        validate(d)
        clock[0] += 10.
        events.append('qualification')

    def transfer(value):
        output = host(value)
        clock[0] += 1.
        events.append('transfer')
        return output

    def measure(model):
        value = metrics(model)
        events.append('measurement')
        value['test_measurement_clock'] = clock[0]
        return value

    monkeypatch.setattr(api, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(api, '_validate_device_fit', final_validation)
    monkeypatch.setattr(api, '_host', transfer)
    monkeypatch.setattr(api, '_device_metrics', measure)
    result = api._export(fitted, source_provenance(), wall_started=0.)
    assert events[0] == 'qualification' and events[-1] == 'measurement'
    phase = result.provenance['phase_seconds']
    assert phase['final_device_qualification_seconds'] == 10.
    assert phase['device_export_seconds'] == events.count('transfer')
    assert result.provenance['through_device_export_seconds'] == clock[0]
    assert result.provenance['test_measurement_clock'] == clock[0]
    assert 'finished_utc' not in result.provenance
    assert np.all(np.isfinite(result.raw_phi))


def test_durable_publication_timing_is_returned_separately_from_receipt(device_reference, monkeypatch, tmp_path):
    import clipp1d.cuda_api as api
    from clipp1d.api import source_provenance
    fitted, data = device_reference
    result = api._export(fitted, source_provenance())
    clock = [100.]
    original_json, original_replace = api._json, api.os.replace

    def receipt_write(path, value):
        clock[0] += 20.
        return original_json(path, value)

    def atomic_publish(source, destination):
        clock[0] += 30.
        original_replace(source, destination)

    monkeypatch.setattr(api, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(api, '_json', receipt_write)
    monkeypatch.setattr(api.os, 'replace', atomic_publish)
    published = api._publish(result, data, tmp_path / 'output', fitted.records, wall_started=0.)
    receipt = json.loads((tmp_path / 'output/run.json').read_text())
    assert receipt['provenance'] == json.loads(json.dumps(published.provenance))
    assert published.provenance['elapsed_seconds'] == 100.
    assert published.operation_metrics['elapsed_seconds'] == 150.
    assert published.operation_metrics['publication_seconds'] == 50.
    assert 'publication_completed_utc' in published.operation_metrics
    assert 'operation_metrics' not in receipt
    assert 'excludes receipt serialization' in receipt['provenance']['elapsed_scope']
    assert set(p.name for p in (tmp_path / 'output').iterdir()) == {
        'run.json', 'mutation_clusters.tsv', 'cluster_centers.tsv', 'mutation_multiplicity.tsv'}
    api._validate_result(published, data, fitted.records)
    published.operation_metrics['elapsed_seconds'] = 0.
    with pytest.raises(api.QualificationError, match='modified scientific provenance'):
        api._validate_result(published, data, fitted.records)


def test_valid_json_receipt_corruption_is_not_published(device_reference, monkeypatch, tmp_path):
    import clipp1d.cuda_api as api
    from clipp1d.api import source_provenance
    fitted, data = device_reference
    result = api._export(fitted, source_provenance())
    original = api._json

    def corrupted_receipt(path, value):
        expected_hash = original(path, value)
        altered = json.loads(path.read_text())
        altered['selected_lambda'] += 1.
        path.write_text(json.dumps(altered))
        return expected_hash

    monkeypatch.setattr(api, '_json', corrupted_receipt)
    destination = tmp_path / 'output'
    with pytest.raises(api.QualificationError, match='receipt readback'):
        api._publish(result, data, destination, fitted.records)
    assert list(destination.iterdir()) == []
    assert list(tmp_path.glob('.clipp1d-publish-*')) == []
