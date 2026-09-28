"""One explicitly allocated GPU: compiled tests and paired observed-input fits.

This controller never initializes CUDA itself, preserving exclusive-process
ownership for sequential fresh children. It submits no scheduler jobs.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def table(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    result = {r['mutation_id']: r for r in rows}
    if len(rows) != len(result) or not rows:
        raise ValueError('Empty/duplicate qualification table')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    if not os.environ.get('LSB_JOBID', '').isdigit():
        raise ValueError('This qualification requires its authorized LSF allocation')
    root = Path(args.outdir).resolve()
    root.mkdir()
    payload = Path(args.inventory).resolve().parent
    inventory = read(args.inventory)
    for name, digest in inventory.items():
        path = payload/name
        if path.is_symlink() or not path.resolve().is_relative_to(payload) or sha(path) != digest:
            raise ValueError('Payload inventory mismatch')
    source = Path(__file__).resolve().parents[1]
    manifest = Path(args.manifest).resolve()
    deadline = time.monotonic()+55*60
    stages, comparison = [], []
    status, error = 'failed', None

    def run(name, command):
        with (root/(name+'.log')).open('x') as log:
            result = subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=max(1, deadline-time.monotonic()))
        stages.append(dict(stage=name, argv=command, returncode=result.returncode,
                           log_sha256=sha(root/(name+'.log'))))
        if result.returncode:
            raise RuntimeError(f'Qualification stage {name} failed')

    try:
        # A separate process proves actual allocation and imports dependencies;
        # it exits before any fitting child initializes its own CUDA allocator.
        probe = (
            'import torch,numpy,pandas,scipy,json,platform; '
            'assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; '
            'x=torch.zeros(1,device="cuda",dtype=torch.float64); torch.cuda.synchronize(); '
            'print(json.dumps(dict(gpu=torch.cuda.get_device_name(0),'
            'torch=torch.__version__,cuda=torch.version.cuda,numpy=numpy.__version__,'
            'pandas=pandas.__version__,scipy=scipy.__version__,python=platform.python_version())))'
        )
        run('allocation', [sys.executable, '-c', probe])
        run('tests', [sys.executable, '-m', 'pytest', '-q', 'tests/test_mixture_cuda.py',
                       '--junitxml='+str(root/'cuda-tests.xml')])
        suites = ET.parse(root/'cuda-tests.xml').getroot()
        nodes = [suites] if suites.tag == 'testsuite' else list(suites.iter('testsuite'))
        totals = {k: sum(int(n.attrib.get(k, 0)) for n in nodes)
                  for k in ('tests', 'failures', 'errors', 'skipped')}
        if totals != dict(tests=3, failures=0, errors=0, skipped=0):
            raise RuntimeError(f'CUDA qualification did not execute all three checks: {totals}')
        runner = str(source/'benchmarks/run_mixture_experiment.py')
        for execution in ('cpu', 'cuda'):
            command = [sys.executable, runner, '--manifest', str(manifest),
                        '--outdir', str(root/execution), '--device', 'cpu' if execution == 'cpu' else 'cuda:0']
            if execution == 'cpu':
                command.append('--cpu-reference')
            run(execution, command)
        cases = read(manifest)['cases']
        for index, case in enumerate(cases):
            cpu, gpu = [root/device/f'{index:05}' for device in ('cpu', 'cuda')]
            receipts = [read(p/'EXPERIMENT.json') for p in (cpu, gpu)]
            for p, receipt in zip((cpu, gpu), receipts):
                for filename, digest in receipt['output_sha256'].items():
                    if sha(p/filename) != digest:
                        raise ValueError('Changed qualification result')
            a, b = [table(p/'mixture_mutation_clusters.tsv') for p in (cpu, gpu)]
            if set(a) != set(b):
                raise ValueError('Qualification population mismatch')
            difference = max(abs(float(a[i]['mixture_ccf'])-float(b[i]['mixture_ccf'])) for i in a)
            if (not math.isfinite(difference) or difference > 1e-7 or
                    any(a[i]['cluster_label'] != b[i]['cluster_label'] or
                        a[i]['multiplicity'] != b[i]['multiplicity'] for i in a) or
                    abs(receipts[0]['score']-receipts[1]['score']) > 1e-6 or
                    receipts[0]['status'] != receipts[1]['status'] or
                    receipts[0]['adaptive'] != receipts[1]['adaptive']):
                raise ValueError(f'Paired experimental output mismatch for {case["case_id"]}')
            comparison.append(dict(case_id=case['case_id'], mutations=len(a), max_ccf_delta=difference,
                                    score_delta=receipts[1]['score']-receipts[0]['score']))
        for name, digest in inventory.items():
            if sha(payload/name) != digest:
                raise ValueError('Payload changed during qualification')
        status = 'passed'
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        write(root/'QUALIFICATION.json', dict(status=status, error=error, stages=stages,
            paired_cases=comparison, inventory_sha256=sha(args.inventory),
            manifest_sha256=sha(manifest), job_id=os.environ['LSB_JOBID'],
            utc=datetime.now(timezone.utc).isoformat(),
            scope='allocated CUDA engineering tests, unmodified complete-graph path, paired component fits',
            full_cohort_accuracy_accepted=False, independent_confirmation_accepted=False,
            production_adopted=False))


if __name__ == '__main__':
    main()
