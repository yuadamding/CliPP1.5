"""Scope the existing A40 qualification to tasks with validated input seeds."""
from pathlib import Path

from common import read, sha


def contract(root, control):
    if control.resolve() != control or not control.is_relative_to(root/'control'):
        raise ValueError('Unexpected expansion generation')
    g = read(control/'GENERATION.json')
    if g['plan_sha256'] != sha(root/'payload/RUN_PLAN.json'):
        raise ValueError('Different expansion plan')
    for name, digest in g['inventory'].items():
        path = control/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or sha(path) != digest:
            raise ValueError('Changed expansion generation')
    admission = read(control/'READY_ADMITTED.json')
    if (admission['plan_sha256'] != g['plan_sha256'] or
            admission['inventory_sha256'] != sha(root/'payload/INVENTORY.json') or
            admission['scope'] != 'validated_existing_seeds_only'):
        raise ValueError('Different existing-seed admission')
    q = Path(admission['qualification_path'])
    if (sha(q) != admission['qualification_sha256'] or
            read(q)['status'] != 'passed' or read(q)['runtime']['gpu'] != 'NVIDIA A40'):
        raise ValueError('Missing original allocated A40 qualification')
    return g


def fresh_seeds_qualified(root):
    q = root/'qualification/lsf/QUALIFIED.json'
    if not q.exists():
        return False
    value = read(q)
    if (value['status'] != 'passed' or value['plan_sha256'] != sha(root/'payload/RUN_PLAN.json') or
            value['inventory_sha256'] != sha(root/'payload/INVENTORY.json')):
        raise ValueError('Different fresh-seed qualification')
    return True


def eligible(task, fresh):
    return not task.get('requires_seed') or fresh
