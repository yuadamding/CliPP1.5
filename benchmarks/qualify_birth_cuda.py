"""One allocated-GPU birth qualification: paired full fits and frozen replay.

This entry point submits no jobs. Execute only inside an authorized allocation;
the replay is discovery evidence and does not replace the held-out study.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from clipp1d.api import source_provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not args.device.startswith('cuda'):
        raise ValueError('Allocated CUDA required')
    # Only sequential child processes initialize CUDA. A controller context
    # would compete with its child on exclusive-process LSF devices.
    args.outdir.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    commands = [
        [sys.executable, str(here/'qualify_partition_search_cuda.py'), '--outdir', str(args.outdir/'full-fit'),
         '--device', args.device],
        [sys.executable, str(here/'replay_birth.py'), '--bundle', str(args.bundle), '--outdir', str(args.outdir/'replay'),
         '--device', args.device],
    ]
    stages = []
    status = 'failed'
    try:
        for index, command in enumerate(commands):
            with (args.outdir/f'stage-{index}.log').open('x') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            stages.append(dict(argv=command, returncode=completed.returncode))
            if completed.returncode:
                raise RuntimeError(f'Qualification stage {index} failed; retained stage log')
        status = 'passed'
    except Exception:
        status = 'failed'
        raise
    finally:
        receipt = dict(status=status, created_utc=datetime.now(timezone.utc).isoformat(),
                       source=source_provenance(), bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
                       stages=stages, scope='allocated CUDA full-fit engineering parity and discovery replay; not held-out accuracy')
        (args.outdir/'QUALIFICATION.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
