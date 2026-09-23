"""Recompute the captured rational obstruction, not just its saved conclusion."""
import importlib.util
import json
from pathlib import Path
import sys

from clipp1d.cuda.policy import CudaPolicy


ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'validation/cuda-bound-recovery-v3'


def test_literal_below256_mathematical_contract_remains_obstructed(tmp_path,monkeypatch):
    script=ARCHIVE/'study/below256_rational_bound.py'
    spec=importlib.util.spec_from_file_location('literal_binary64_obstruction',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output=tmp_path/'proof.json'
    monkeypatch.setattr(sys,'argv',[str(script),'--repository-root',str(ROOT),
        '--evidence-root',str(ARCHIVE),'--output',str(output)])
    module.main()
    actual=json.loads(output.read_bytes())
    expected=json.loads((ARCHIVE/'study/BELOW256_RATIONAL_BOUND_V3.json').read_bytes())
    assert actual==expected
    assert actual['status']=='exact_mathematical_gate_incompatibility_proved'
    assert min(r['gap_cost_to_global_allowance']['approximate'] for r in actual['adjacent_floats_bracketing_stationary_root'])>241
    assert CudaPolicy().inner_atol==1e-10 and CudaPolicy().inner_rtol==1e-11
    assert CudaPolicy().inner_kkt_tol==1e-7
