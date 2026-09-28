"""Admission and publication failures must block the expanded experiment."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

OPS = Path(__file__).resolve().parents[1]/'benchmarks/mixture_lsf'
# Load an isolated common module; other studies have their own module of this name.
def load(name):
    spec = importlib.util.spec_from_file_location('mixture_ops_'+name, OPS/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load('common')
saved = sys.modules.get('common')
sys.modules['common'] = common
try:
    controller, worker = load('controller'), load('worker')
finally:
    if saved is None:
        del sys.modules['common']
    else:
        sys.modules['common'] = saved


def admission():
    return ('Job <123>, Job Name <study_00000>, User <yding4>, Status <PSUSP>, '
        'Queue <egpu>, Command <worker>, Requested Resources <rusage[mem=32] span[hosts=1]>, '
        'Requested GPU <num=1:mode=exclusive_process:gmodel=NVIDIAL40>\n'
        ' RUNLIMIT\n 60.0 min\n MEMLIMIT\n 32 G\n')


@pytest.mark.parametrize('model', ['NVIDIAL40', 'NVIDIAA40'])
def test_exact_cpu_memory_gpu_admission(model):
    a = dict(job_id='123', job_name='study_00000', wall_minutes=60, gpu_model=model)
    raw = admission().replace('NVIDIAL40', model)
    assert controller.verify_admission(raw, a, 'worker') == 'PSUSP'
    for old, new in [('32 G', '64 G'), ('60.0 min', '600.0 min'), (model, 'TeslaV100'),
                     ('User <yding4>', 'User <someone>'),
                     (', Requested Resources', ', 2 Task(s), Requested Resources')]:
        with pytest.raises(ValueError):
            controller.verify_admission(raw.replace(old, new), a, 'worker')


def test_uncertain_submit_is_durable_and_cannot_be_repeated(tmp_path, monkeypatch):
    (tmp_path/'receipts').mkdir()
    common.write(tmp_path/'receipts/qualification-admitted.json', dict(gpu_model='NVIDIAA40'))
    (tmp_path/'payload').mkdir()
    common.write(tmp_path/'payload/RUN_PLAN.json', {})
    calls = []
    def command(argv, *args):
        calls.append(argv)
        return dict(code=None, stdout='Job <123> is submitted to queue <egpu>.', stderr='timeout')
    monkeypatch.setattr(controller, 'command', command)
    task = dict(key='00000')
    p = dict(job_prefix='study_', python='/python')
    with pytest.raises(RuntimeError, match='Ambiguous'):
        controller.submit(tmp_path, p, task, 60)
    assert common.read(tmp_path/'receipts/00000/accepted.json')['job_id'] == '123'
    with pytest.raises(FileExistsError):
        controller.submit(tmp_path, p, task, 60)
    assert len(calls) == 1
    assert calls[0][calls[0].index('-gpu')+1].endswith('gmodel=NVIDIAA40')


@pytest.mark.parametrize('gpu,model', [('NVIDIA A40', 'NVIDIAA40'), ('NVIDIA L40', 'NVIDIAL40')])
def test_follow_on_model_requires_hash_bound_actual_allocation(tmp_path, gpu, model):
    (tmp_path/'output').mkdir()
    log = tmp_path/'output/allocation.log'
    log.write_text(json.dumps(dict(gpu=gpu))+'\n')
    q = dict(stages=[dict(stage='allocation', returncode=0, log_sha256=common.sha(log))])
    assert common.qualified_gpu_model(tmp_path, q) == model
    log.write_text(json.dumps(dict(gpu='NVIDIA H100'))+'\n')
    with pytest.raises(ValueError, match='Unbound'):
        common.qualified_gpu_model(tmp_path, q)
    q['stages'][0]['log_sha256'] = common.sha(log)
    with pytest.raises(ValueError, match='Unsupported'):
        common.qualified_gpu_model(tmp_path, q)


def test_runtime_sizing_never_changes_scientific_budget_or_silently_clips():
    c = [dict(elapsed_seconds=80, retained_mutations=100)]
    assert controller.sizing(dict(retained_mutations=50), c) == 10
    assert controller.sizing(dict(retained_mutations=500), c) == 34
    with pytest.raises(ValueError, match='ceiling'):
        controller.sizing(dict(retained_mutations=5000), c)


def test_publication_canary_rejects_label_ccf_and_missing_rows(tmp_path):
    expected = tmp_path/'payload/expected/00000'
    actual = tmp_path/'outputs/00000/guarded/00000'
    expected.mkdir(parents=True)
    actual.mkdir(parents=True)
    name = 'mixture_mutation_clusters.tsv'
    text = 'mutation_id\tcluster_label\tmixture_ccf\tmultiplicity\na\t0\t.9\t1\nb\t1\t.4\t2\n'
    (expected/name).write_text(text)
    common.write(expected/'EXPECTATION.json', dict(table_sha256=common.sha(expected/name),ccf_atol=1e-7,
        status='preserved_baseline', original_baseline_preserved=True, selected_family='preserved_complete_graph_partition'))
    common.write(actual/'EXPERIMENT.json', dict(status='preserved_baseline',original_baseline_preserved=True,
        structural_decision=dict(selected_family='preserved_complete_graph_partition')))
    (actual/name).write_text(text)
    assert worker.check_publication_canary(tmp_path, dict(key='00000'))['status'] == 'passed'
    for wrong in (text.replace('b\t1', 'b\t0'), text.replace('.4', 'nan'), text.split('b\t')[0]):
        (actual/name).write_text(wrong)
        with pytest.raises(ValueError):
            worker.check_publication_canary(tmp_path, dict(key='00000'))


def test_missing_scheduler_job_retains_unknown_slot(monkeypatch):
    monkeypatch.setattr(controller, 'command', lambda *a: dict(code=255,stdout='',stderr='network'))
    assert controller.scheduler_state(dict(job_id='123',job_name='study'))[0] == 'UNKNOWN'
    monkeypatch.setattr(controller, 'command', lambda *a: dict(code=0,stdout='123 other RUN study',stderr=''))
    with pytest.raises(ValueError, match='identity'):
        controller.scheduler_state(dict(job_id='123',job_name='study'))


def test_no_clobber_receipt_preserves_original(tmp_path):
    p = tmp_path/'receipt.json'
    common.write(p, dict(a=1))
    with pytest.raises(FileExistsError):
        common.write(p, dict(a=2))
    assert json.loads(p.read_text()) == dict(a=1)


def test_aged_out_job_requires_its_exact_scheduler_report(tmp_path):
    path = tmp_path/'lsf.out'
    a = dict(job_id='123', job_name='study')
    assert controller.terminal_log_state(path, a) is None
    path.write_text('Successfully completed.\n')
    assert controller.terminal_log_state(path, a) is None
    raw = ('Subject: Job 123: <study>\nJob <study> was submitted today by user <yding4>\n'
           'Results reported at now\nTerminated at now\nSuccessfully completed.\n')
    path.write_text(raw)
    assert controller.terminal_log_state(path, a) == 'DONE'
    path.write_text(raw.replace('Job 123:', 'Job 456:'))
    assert controller.terminal_log_state(path, a) is None


def test_validation_requires_complete_source_tables_population_and_centers(tmp_path):
    root = tmp_path
    payload = root/'payload'
    (payload/'manifests').mkdir(parents=True)
    inventory = {}
    for name in ('src/clipp1d/model.py', 'benchmarks/run_mixture_experiment.py',
                 'benchmarks/mixture_structural_guard.py', 'benchmarks/apply_mixture_structural_guard.py'):
        path = payload/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# frozen fixture\n')
        inventory['source/'+name] = common.sha(path)
    common.write(payload/'INVENTORY.json', inventory)
    inp, seed = root/'input.tsv', root/'seed.tsv'
    inp.write_text('bound input')
    seed.write_text('mutation_id\na\nb\n')
    case = dict(case_id='case', input_path=str(inp), input_sha256=common.sha(inp),
                seed_path=str(seed), seed_sha256=common.sha(seed))
    manifest = payload/'manifests/00000.json'
    common.write(manifest, dict(cases=[case], policy={}))
    task = dict(key='00000', case_id='case', retained_mutations=2,
                manifest='manifests/00000.json', manifest_sha256=common.sha(manifest))
    receipts = {}
    for mode in ('mixture', 'guarded'):
        folder = root/'outputs/00000'/mode
        output = folder/'00000'
        output.mkdir(parents=True)
        (output/'mixture_mutation_clusters.tsv').write_text(
            'mutation_id\tcluster_label\tmixture_ccf\tmultiplicity\na\t0\t.9\t1\nb\t1\t.4\t2\n')
        (output/'mixture_cluster_centers.tsv').write_text(
            'cluster_label\tmixture_ccf\tn_mutations\n0\t.9\t1\n1\t.4\t1\n')
        source = {name.removeprefix('source/'): v for name, v in inventory.items()
                  if mode == 'guarded' or name.endswith(('model.py', 'run_mixture_experiment.py'))}
        binding = dict(policy={},manifest_sha256=common.sha(manifest),source_inventory=source,
            execution='compiled_cuda',selector_execution='compiled_cuda',selection_policy='guard')
        common.write(folder/'BINDING.json', binding)
        common.write(folder/'COMPLETE.json', dict(source_unchanged=True,manifest_unchanged=True,
            failures=[],results=[dict(case_id='case',output=str(output))]))
        receipt = dict(inputs=case,policy={},case_id='case',device='cuda:0',
            selector_execution='compiled_cuda',selection_policy='guard',
            output_sha256={p.name:common.sha(p) for p in output.iterdir()})
        if mode == 'guarded':
            parent = root/'outputs/00000/mixture/00000/EXPERIMENT.json'
            receipt.update(parent_receipt=str(parent),parent_receipt_sha256=common.sha(parent))
        common.write(output/'EXPERIMENT.json', receipt)
        receipts[mode] = output/'EXPERIMENT.json'
    assert set(common.validate_outputs(root, task)) == {'mixture', 'guarded'}
    receipt = common.read(receipts['guarded'])
    original = receipts['guarded'].read_text()
    receipt['output_sha256'] = {}
    receipts['guarded'].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='contract'):
        common.validate_outputs(root, task)
    receipts['guarded'].write_text(original)
    path = receipts['guarded'].parent/'mixture_mutation_clusters.tsv'
    path.write_text(path.read_text().replace('b\t1\t.4', 'b\t1\t.5'))
    receipt = json.loads(original)
    receipt['output_sha256'][path.name] = common.sha(path)
    receipts['guarded'].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='Center table'):
        common.validate_outputs(root, task)
