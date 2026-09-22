import csv
import io
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from repair_simclone_normal_cn import repair


def test_repair_changes_only_simclone_normal_cn_and_preserves_original(tmp_path):
    text = ('##source_cohort=SimClone1000\n'
            'mutation_id\tchromosome\tnormal_cn\talt_count\tallele_a_cn\n'
            'a\t1\t2\t5\t1\nx\tX\t1\t12\t3\ny\tY\t1\t9\t1\n')
    source, target = tmp_path/'source.tsv', tmp_path/'corrected.tsv'
    source.write_text(text)
    result = repair(source, target)
    assert result['changed_mutations'] == 2
    assert source.read_text() == text
    def table(value):
        return list(csv.DictReader(io.StringIO(''.join(line for line in value.splitlines(True)
                                                       if not line.startswith('#'))), delimiter='\t'))
    before, after = table(text), table(target.read_text())
    for a, b in zip(before, after):
        assert b['normal_cn'] == '2'
        assert {k:v for k,v in a.items() if k != 'normal_cn'} == {k:v for k,v in b.items() if k != 'normal_cn'}
    with pytest.raises(FileExistsError):
        repair(source, target)
    source.write_text(text.replace('SimClone1000', 'PhylogicNDT500'))
    with pytest.raises(ValueError, match='SimClone1000'):
        repair(source, tmp_path/'other.tsv')
