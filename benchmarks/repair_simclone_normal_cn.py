"""Repair canonical SimClone inputs using the simulator's normal-CN=2 contract.

Original inputs are preserved. This rule is specific to SimClone; it is not a
sex-chromosome rule for real tumors or other simulators. See the source audit in
RESEARCH_CHAIN_PROPOSALS.md and the immutable study receipts linked there.
"""
import csv
import hashlib
from io import StringIO
from pathlib import Path


def repair(source, destination):
    source, destination = Path(source), Path(destination)
    payload = source.read_text()
    lines = payload.splitlines(keepends=True)
    comments = [line for line in lines if line.startswith('#')]
    if '##source_cohort=SimClone1000\n' not in comments:
        raise ValueError('Repair requires explicit SimClone1000 source metadata')
    rows = list(csv.DictReader((line for line in lines if not line.startswith('#')), delimiter='\t'))
    changed = []
    for row in rows:
        normal = float(row['normal_cn'])
        chrom = row['chromosome'].removeprefix('chr').upper()
        if normal != 2:
            if chrom not in ('X', 'Y') or normal not in (0, 1):
                raise ValueError('Unexpected normal CN; refusing an unqualified repair')
            changed.append(row['mutation_id'])
            row['normal_cn'] = '2'
    stream = StringIO()
    stream.writelines(comments)
    stream.write('##normal_cn_contract=SimClone_simulator_diploid_normal_all_chromosomes\n')
    stream.write('##original_input_sha256=' + hashlib.sha256(source.read_bytes()).hexdigest() + '\n')
    writer = csv.DictWriter(stream, list(rows[0]), delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    with destination.open('x') as handle:
        handle.write(stream.getvalue())
    return {'changed_mutations': len(changed), 'changed_ids': changed,
            'original_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'corrected_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'rule': 'SimClone normal_cn=2 on every chromosome; counts and tumor CN preserved'}
