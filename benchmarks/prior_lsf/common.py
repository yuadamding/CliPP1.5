"""Receipt and identity primitives for the frozen single-tumor CUDA cohort."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value, replace=False):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('xb') as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)
    else:
        os.link(temporary, path)
        temporary.unlink()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def command(argv, timeout=60):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return dict(code=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired as error:
        def text(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
        return dict(code=None, stdout=text(error.stdout), stderr=text(error.stderr), error='timeout')


def identity(pid):
    proc = Path('/proc') / str(pid)
    fields = (proc/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid, start_ticks=fields[19], pgrp=int(fields[2]), session=int(fields[3]),
                tty=int(fields[4]), cmdline_sha256=digest(proc/'cmdline'), host=os.uname().nodename)


def plan(root):
    assert root.resolve() == root and root.stat().st_uid == os.getuid() == 307469
    p = read(root/'PREPARED.json')
    assert str(root) == p['remote_root'] and p['max_submitted'] == 1
    return p


def verify_files(root, full=False):
    inventory = read(root/'inventory.json')
    for name, expected in inventory.items():
        # Blob fully verified at publication and controller admission; workers
        # verify their own ZIP members, plus all executable/configuration bytes.
        if name.startswith('datasets/') and not full:
            continue
        path = root/name
        assert path.is_file() and not path.is_symlink() and digest(path) == expected, name


def accepted(root, key):
    folder = root/'receipts'/key
    a, d = read(folder/'accepted.json'), read(folder/'admitted.json')
    assert a['job_id'] == d['job_id'] == os.environ['LSB_JOBID']
    assert a['key'] == key and a['plan_sha256'] == digest(root/'PREPARED.json')
    assert a['job_name'] == plan(root)['job_prefix'] + key
    return a, folder


def cases(root):
    fixed = read(root/'cases.json')
    dynamic = root/'receipts/factorial-cases.json'
    return fixed + (read(dynamic)['cases'] if dynamic.exists() else [])
