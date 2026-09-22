"""Replay the eight original source-bound cases against the frozen prototype."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from benchmark_chain import configure_execution
CONTROLS=configure_execution(1,threads=1)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from clipp1d import fit  # noqa: E402
from clipp1d.api import source_provenance  # noqa: E402
from clipp1d.report import write_json  # noqa: E402

root=Path(__file__).resolve().parents[1]
study=root/'results/clipp2-transfer-study-20260921-v1'
out=root/'results/single-region-four-cohorts-20260921-v1/integration-qualification'
out.mkdir()
records=[]
for folder in ['simclone-matched-20260921-v2','simclone-additional-four-20260921-v1']:
    parent=root/'results'/folder
    for case in json.loads((parent/'plan.json').read_text())['cases']:
        name=case['tumor_id']
        expected=json.loads((study/'best-seed-known-results'/f'{name}.json').read_text())
        result=fit(parent/name/'input.tsv',out/name)
        variant=expected['variants']['ward_and_boundary']
        # The prior script stores estimator fields under result and provenance under diagnostics.
        candidate=variant.get('result',variant)
        if 'cuts' not in candidate:
            candidate=variant['refit']
        assert list(result.partition)==candidate['cuts'],name
        assert abs(result.selection_score-candidate['score'])<1e-8,name
        assert result.candidate_provenance['raw_reference']['refit_score'] >= result.selection_score-1e-8
        assert result.provenance['input_sha256']==expected['input_sha256']
        records.append(dict(case_id=name,score=result.selection_score,
                            direct=result.selected_lambda is None,seconds=result.provenance['elapsed_seconds']))
        print(records[-1],flush=True)
write_json(out/'qualification.json',dict(status='passed',cases=records,controls=CONTROLS,source=source_provenance()))
