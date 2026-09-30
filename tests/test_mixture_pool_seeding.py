"""Metadata-only checks for truth-free, source-bound baseline preparation."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

OPS = Path(__file__).resolve().parents[1]/'benchmarks/mixture_pool'


def load(name):
    spec = importlib.util.spec_from_file_location('seed_test_'+name, OPS/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load('common')
previous = sys.modules.get('common')
try:
    sys.modules['common'] = common
    seeding = load('seeding')
finally:
    if previous is None:
        del sys.modules['common']
    else:
        sys.modules['common'] = previous


@pytest.fixture
def generated(tmp_path, monkeypatch):
    from clipp1d import api, cuda_api
    from clipp1d.cuda.policy import CudaPolicy
    from clipp1d.cuda.refinement import PartitionSearchPolicy

    (tmp_path/'payload').mkdir()
    source = tmp_path/'input.tsv'
    source.write_text('metadata fixture; no numerical fit is run\n')
    request = dict(schema='clipp1d.mixture_seed_request.v1', case_id='case',
        input_path=str(source), input_sha256=common.sha(source), policy={'bound': True},
        partition_policy=asdict(PartitionSearchPolicy()), cuda_policy=asdict(CudaPolicy()),
        source_sha256='source')
    common.write(tmp_path/'payload/request.json', request)
    task = dict(key='case', case_id='case', manifest='request.json',
        manifest_sha256=common.sha(tmp_path/'payload/request.json'), retained_mutations=2,
        retained_ids_sha256=hashlib.sha256(json.dumps(['a', 'b']).encode()).hexdigest())
    monkeypatch.setattr(api, 'source_provenance', lambda: {'source_sha256': 'source'})

    def fake_fit(path, destination, **kwargs):
        assert path == str(source)
        assert kwargs['device'] == 'cuda:0'
        assert kwargs['partition_search'] and not kwargs['generic_partition_grouping']
        assert asdict(kwargs['partition_policy']) == request['partition_policy']
        destination.mkdir()
        text = ('mutation_id\tstatus\tcluster_label\tpartition_ccf\tmultiplicity_call\n'
                'a\tretained\t0\t0.9\t1\n'
                'b\tretained\t1\t0.3\t2\n'
                'excluded\tMAJOR_CN_ABOVE_LIMIT\tNA\tNA\tNA\n')
        names = ['mutation_clusters.tsv', 'mutation_multiplicity.tsv', 'cluster_centers.tsv',
                 'partition_mutation_clusters.tsv', 'partition_cluster_centers.tsv']
        for name in names:
            (destination/name).write_text(text if name == 'partition_mutation_clusters.tsv' else 'fixture\n')
        provenance = dict(input_sha256=request['input_sha256'], source_sha256='source',
            backend='cuda', dtype='float64', compiled_inference=True, cpu_numeric_fallback=False,
            clonal_constraint=False, max_major_cn=4, policy=request['cuda_policy'],
            partition_search=request['partition_policy'])
        common.write(destination/'run.json', dict(schema='clipp1d.cuda.run.v5', status='success',
            provenance=provenance, table_sha256={n: common.sha(destination/n) for n in names}))

    monkeypatch.setattr(cuda_api, 'fit', fake_fit)
    output = tmp_path/'output'
    output.mkdir()
    seeding.generate(tmp_path, task, output)
    return tmp_path, task, output


def test_generated_manifest_excludes_filtered_mutations_and_keeps_policy(generated):
    root, task, output = generated
    manifest = seeding.prepared_manifest(root, task, output)
    assert [r['mutation_id'] for r in seeding.check_seed(output/'seed.tsv', task)] == ['a', 'b']
    assert common.read(manifest)['policy'] == {'bound': True}
    assert common.read(output/'SEED_COMPLETE.json')['truth_used'] is False


@pytest.mark.parametrize('name', ['seed.tsv', 'INFERENCE.json', 'baseline/run.json',
                                 'baseline/partition_mutation_clusters.tsv'])
def test_generated_manifest_rejects_changed_ancestry(generated, name):
    root, task, output = generated
    with (output/name).open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='Changed'):
        seeding.prepared_manifest(root, task, output)


def test_seed_request_rejects_truth_parameters(generated):
    root, task, _ = generated
    path = root/'payload/request.json'
    value = common.read(path)
    value['truth_path'] = 'not allowed'
    path.write_text(json.dumps(value))
    task['manifest_sha256'] = common.sha(path)
    with pytest.raises(ValueError, match='scientific policy'):
        seeding.request(root, task)
