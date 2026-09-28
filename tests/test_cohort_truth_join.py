import csv
import json
from pathlib import Path
import sys

import pytest

from clipp1d.io import SCHEMA_COLUMNS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from prepare_four_cohorts import TRUTH_SCHEMA, coord, phylogic_mixed_cn, truth_recoverable, unique


def test_truth_coordinate_notation_and_conflicts():
    assert coord(dict(c='chr1', p='5.8e+07'), 'c', 'p') == 'chr1:58000000'
    with pytest.raises(ValueError):
        coord(dict(c='1', p='5.8'), 'c', 'p')
    rows = [dict(c='1', p='20', value='a'), dict(c='1', p='20', value='b'),
            dict(c='1', p='20', value='a'), dict(c='2', p='20', value='a')]
    result = unique(rows, 'c', 'p', ['value'])
    assert result == {'chr1:20': None, 'chr2:20': ('a',)}


def test_original_mixed_cn_is_numeric_and_requires_two_occupied_states():
    assert phylogic_mixed_cn('2', '1', '.4', '1', '1')
    assert not phylogic_mixed_cn('2.0', '1', '.4', '2', '1.00')
    assert not phylogic_mixed_cn('2', '1', '1', 'NA', 'NA')
    assert not phylogic_mixed_cn('2', '1', '0', '1', '1')
    for bad in ('nan', '-.1', '1.1'):
        with pytest.raises(ValueError, match='fraction'):
            phylogic_mixed_cn('2', '1', bad, '1', '1')
    with pytest.raises(ValueError, match='integer CN'):
        phylogic_mixed_cn('2', '1', '.4', '1.5', '1')


def test_legacy_phylogic_truth_requires_new_versioned_preparation():
    old = dict(dataset='PhylogicNDT500_TSV', status='prepared')
    assert not truth_recoverable(old)
    assert truth_recoverable(dict(old, truth_schema=TRUTH_SCHEMA))
    assert truth_recoverable(dict(dataset='SimClone1000_TSV'))


def test_phylogic_preparation_serializes_original_mixture_flags(tmp_path, monkeypatch):
    import prepare_four_cohorts as prep

    def write_table(path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, records[0], delimiter='\t')
            writer.writeheader()
            writer.writerows(records)

    inputs, truth, assignments = [], [], []
    for i, (major, fraction, major2) in enumerate([(2, .4, 1), (2, .4, '2.0'), (5, .4, 1), (2, 1, 'NA')]):
        row = dict(zip(SCHEMA_COLUMNS, [f'm{i}', '01', 10, 90, 1, .8, 2,
                                       f'seg{i}', 's1', 1, major, 1]))
        inputs.append(dict(row, chromosome='1', position=i+1))
        truth.append(dict(Chromosome='1', Start_position=i+1, mut_ccf=.6,
                          integer_mult=1, mult=1, t_alt_count=10, t_ref_count=90,
                          maj_a=major, min_a=1, frac_s=fraction,
                          nMaj2_A=major2, nMin2_A=1 if fraction < 1 else 'NA'))
        assignments.append(dict(chr='1', pos=i+1, cluster=2))
    source = tmp_path/'input.tsv'
    write_table(source, inputs)
    simulator = tmp_path/'original'/'training_PhylogicNDT500_simulations_12_16'
    write_table(simulator/'Mafs_Real_Info'/'case.truth_vals.maf.tsv', truth)
    write_table(simulator/'Mut_Assign'/'case.mutation_assignments.txt', assignments)
    monkeypatch.setattr(prep, 'SIM', tmp_path/'original')
    original = source.read_bytes()
    result = prep.prepare(dict(dataset='PhylogicNDT500_TSV', case_id='case',
                               input_path=str(source), expected_sha256=prep.sha(source)),
                          tmp_path/'prepared')
    table = prep.rows(result['truth_path'])
    assert {r['mutation_id']: int(r['mixed_cn']) for r in table} == {'m0': 1, 'm1': 0, 'm3': 0}
    assert result['truth_mixed_cn_mutations'] == 2  # Includes the excluded major-CN-five row.
    assert result['truth_mixed_cn_retained'] == 1
    assert result['truth_schema'] == TRUTH_SCHEMA
    assert result['truth_coverage'] == 1
    assert source.read_bytes() == original
    receipt = json.loads((Path(result['truth_path']).parent/'preparation.json').read_text())
    assert receipt == result
    with pytest.raises(FileExistsError):
        prep.prepare(result, tmp_path/'prepared')
