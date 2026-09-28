"""One immutable claim per case across LSF and both Kubernetes device pools."""
from common import now, read, sha, write


def claim(root, owner):
    if (root/'STOP').exists():
        return None
    imported = read(root/'payload/IMPORTED.json')
    for task in read(root/'payload/UNITS.json'):
        key = task['key']
        if key in imported:
            continue
        if (root/'STOP').exists():
            return None
        directory = root/'claims'/key
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        # A crash between mkdir and publication retains the claim, never retries.
        write(directory/'claim.json', dict(key=key, case_id=task['case_id'],
              owner=owner, plan_sha256=sha(root/'payload/RUN_PLAN.json'), utc=now()))
        return task
    return None
