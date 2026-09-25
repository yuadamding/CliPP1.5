"""Check scalar-grid sensitivity where oracle/external partitions did not win."""
from concurrent.futures import ProcessPoolExecutor
import json
import pandas as pd
from diagnose import H, diagnose, sha


def main():
    previous = pd.read_csv(H/'all/per_case_candidates.tsv', sep='\t')
    target = previous[(previous.candidate.isin(['truth_partition_approx_refit', 'pyclone_partition_approx_refit']))
                      & ~previous.lower_score_than_published]
    ids = sorted(target.case_id.unique())
    output = H/'grid4097_verification'
    output.mkdir(exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for record in pool.map(diagnose, ids, [4097]*len(ids)):
            (output/(record['case_id']+'.json')).write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
            for row in record['rows']:
                old = previous[(previous.case_id == record['case_id']) & (previous.candidate == row['candidate'])].iloc[0]
                rows.append(dict(case_id=record['case_id'], candidate=row['candidate'],
                                 score1025=float(old.score), score4097=row['score'],
                                 difference=row['score']-float(old.score)))
    table = pd.DataFrame(rows)
    table.to_csv(output/'comparison.tsv', sep='\t', index=False)
    summary = dict(cases=len(ids), maximum_processes=4,
                   selection='union of cases where truth or PyClone partition failed to lower published score on 1025-grid diagnostic',
                   maximum_absolute_score_difference=float(table.difference.abs().max()),
                   script_sha256=sha(__file__), diagnostic_script_sha256=sha(H/'diagnose.py'),
                   limitations='Grid agreement is a sensitivity check, not a global scalar certificate.')
    (output/'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
