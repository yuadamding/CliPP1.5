from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from prepare_four_cohorts import coord, unique


def test_truth_coordinate_notation_and_conflicts():
    assert coord(dict(c='chr1', p='5.8e+07'), 'c', 'p') == 'chr1:58000000'
    with pytest.raises(ValueError):
        coord(dict(c='1', p='5.8'), 'c', 'p')
    rows = [dict(c='1', p='20', value='a'), dict(c='1', p='20', value='b'),
            dict(c='1', p='20', value='a'), dict(c='2', p='20', value='a')]
    result = unique(rows, 'c', 'p', ['value'])
    assert result == {'chr1:20': None, 'chr2:20': ('a',)}
