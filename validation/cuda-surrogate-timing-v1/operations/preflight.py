"""Qualify one new immutable synthetic-study transfer destination; no jobs."""
from pathlib import Path
import re
import sys

sys.path.insert(0, '/storage/CliPP2/scripts')
import stage_sharedcn_lsf20 as s

s.HOP = s.SSH[:2] + ['-F', '/Users/yding4/.ssh/config.d/codex-chain/company-seadragon.conf'] + s.SSH[2:] + ['-o', 'HostName=ldragon5', '-o', 'HostKeyAlias=seadragon', 'seadragon']
study = Path(__file__).resolve().parent
name = sys.argv[1]
assert re.fullmatch(r'[a-z][a-z0-9-]*-[a-z][a-z0-9]*', name)
parent = study / name
parent.mkdir()
s.preflight(parent, transfer_prefix='clipp2_clipp1d_timing_transfer_20260923-')
