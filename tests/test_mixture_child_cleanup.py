"""Fit exit, timeout and orphaned compiler-helper cleanup use real processes."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

OPS = Path(__file__).resolve().parents[1]/'benchmarks/mixture_pool'
spec = importlib.util.spec_from_file_location('mixture_child_common', OPS/'common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
saved = sys.modules.get('common')
try:
    sys.modules['common'] = common
    spec = importlib.util.spec_from_file_location('mixture_child_helpers', OPS/'case_helpers.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
finally:
    if saved is None:
        del sys.modules['common']
    else:
        sys.modules['common'] = saved


@pytest.mark.parametrize('leader_code', [0, 7])
def test_exited_leader_does_not_leave_live_or_zombie_descendants(tmp_path, leader_code):
    grandchild = tmp_path/'grandchild.py'
    marker = tmp_path/'ready'
    grandchild.write_text('import os, signal, time\nfrom pathlib import Path\n'
        'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
        f'Path({str(marker)!r}).write_text(str(os.getpid()))\n'
        'time.sleep(60)\n')
    leader = tmp_path/'leader.py'
    leader.write_text('import subprocess, sys, time\nfrom pathlib import Path\n'
        f'subprocess.Popen([sys.executable, {str(grandchild)!r}])\n'
        f'while not Path({str(marker)!r}).exists(): time.sleep(.01)\n'
        f'raise SystemExit({leader_code})\n')
    log = tmp_path/'child.log'
    try:
        if leader_code:
            with pytest.raises(RuntimeError, match='Child exited 7'):
                helpers.run_child([sys.executable, str(leader)], dict(os.environ), log, time.monotonic()+10)
        else:
            helpers.run_child([sys.executable, str(leader)], dict(os.environ), log, time.monotonic()+10)
        pid = int(marker.read_text())
        assert not Path(f'/proc/{pid}').exists()
        receipt = json.loads(log.read_text().splitlines()[-1])['child_group_cleanup']
        assert receipt['group_absent'] and signal.SIGKILL in receipt['signals']
        assert receipt['leader_returncode'] == leader_code
    finally:
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_timeout_cleans_group_and_preserves_timeout(tmp_path):
    log = tmp_path/'timeout.log'
    with pytest.raises(subprocess.TimeoutExpired):
        helpers.run_child([sys.executable, '-c', 'import time; time.sleep(60)'],
                          dict(os.environ), log, time.monotonic()+.05)
    receipt = json.loads(log.read_text())['child_group_cleanup']
    assert receipt['group_absent'] and receipt['leader_returncode'] < 0
