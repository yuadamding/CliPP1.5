"""Controlled CN-first draws; separate fitting and evaluator-only inventories."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from clipp1d.io import SCHEMA_COLUMNS, read_tumor
from clipp1d.model import compile_model
from clipp1d.simulation.generate_clippsim4k import sample_copy_numbers, sample_multiplicity

DESIGNS = (
    ((1.,), (400,)),
    ((1., .4), (240, 160)),
    ((1., .65, .3), (200, 120, 80)),
    ((1., .75, .5, .25), (160, 100, 80, 60)),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def write_case(root, case_id, alt, ref, major, minor, purity, labels, ccf, multiplicity):
    path = root/'inputs'/f'{case_id}.tsv'
    truth = root/'truth'/f'{case_id}.tsv'
    with path.open('x', newline='') as handle, truth.open('x', newline='') as target:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        evaluator = csv.writer(target, delimiter='\t', lineterminator='\n')
        writer.writerow(SCHEMA_COLUMNS)
        evaluator.writerow(['mutation_id', 'cluster_id', 'ccf', 'multiplicity', 'major_cn', 'minor_cn'])
        for i in range(len(alt)):
            mid = f'm{i:06}'
            writer.writerow([mid, 's1', int(alt[i]), int(ref[i]), 1, purity, 2,
                             f'seg{i}', 'state1', 1., int(major[i]), int(minor[i])])
            evaluator.writerow([mid, int(labels[i]), float(ccf[i]), int(multiplicity[i]),
                                int(major[i]), int(minor[i])])
    model = compile_model(read_tumor(path))
    if len(model) != len(alt):
        raise ValueError('Controlled input has an unexpected exclusion')
    return dict(case_id=case_id, input_file=str(path.relative_to(root)), input_sha256=sha(path),
                n=len(alt)), dict(truth_file=str(truth.relative_to(root)), truth_sha256=sha(truth))


def generate_panel(root, stage):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root/'inputs').mkdir()
    (root/'truth').mkdir()
    seed, replicates = {'development': (2026092601, 1), 'confirmation': (2026092602, 3)}[stage]
    rng = np.random.RandomState(seed)
    cells = list(itertools.product((100, 200, 500), (.4, .6, .9), (.1, .4, .7), range(4), range(replicates)))
    # Blind IDs do not encode true K, CCFs, labels or multiplicity.
    rng.shuffle(cells)
    fit_cases, truth_cases = [], []
    for index, (depth, purity, cna, design_index, replicate) in enumerate(cells):
        centers, counts = DESIGNS[design_index]
        labels = np.repeat(np.arange(len(counts)), counts)
        rng.shuffle(labels)
        ccf = np.asarray(centers)[labels]
        major, minor = sample_copy_numbers(rng, 400, cna)
        multiplicity = sample_multiplicity(rng, major)
        coverage = rng.poisson(depth, size=400)
        vaf = purity*ccf*multiplicity/(2*(1-purity)+purity*(major+minor))
        alt = rng.binomial(coverage, vaf)
        case_id = f'{stage}-{index:04}'
        fitted, truth = write_case(root, case_id, alt, coverage-alt, major, minor, purity, labels, ccf, multiplicity)
        fit_cases.append(fitted)
        truth_cases.append(dict(fitted, **truth, depth=depth, purity=purity, cna_rate=cna,
                               design_index=design_index, true_k=len(counts), replicate=replicate,
                               centers=centers, sizes=counts))
    write_json(root/'FIT_MANIFEST.json', dict(schema='clipp1d.prior_panel.inputs.v1', stage=stage, cases=fit_cases,
        truth_in_fit_manifest=False, generator_sha256=sha(__file__)))
    write_json(root/'EVALUATION_MANIFEST.json', dict(schema='clipp1d.prior_panel.truth.v1', stage=stage,
        generation_seed=seed, rng='numpy RandomState MT19937', cases=truth_cases,
        observation_law='CN first, multiplicity uniform 1..major, Poisson depth, binomial alt',
        fixed_controlled_design=True, unchanged_default_simulator=False,
        generator_sha256=sha(__file__), fitting_manifest_sha256=sha(root/'FIT_MANIFEST.json')))
    return len(fit_cases)


def generate_mechanisms(root):
    """Small engineering fixtures; their truth is only in the evaluator inventory."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root/'inputs').mkdir()
    (root/'truth').mkdir()
    specs = [
        ('no_ambiguity_one', [1]*8, [1]*8, [1.]*8, .8, 400),
        ('no_ambiguity_multi', [1]*8, [1]*8, [1.]*4+[.4]*4, .8, 400),
        ('two_aliases', [2], [0], [.75], .8, 400),
        ('anchored', [2]*4+[1]*4, [0]*4+[1]*4, [.75]*8, .8, 400),
        ('false_split_one', [2, 3, 4, 2]*3, [0, 1, 2, 2]*3, [1.]*12, .8, 400),
        ('false_split_point6', [2, 3, 4, 2]*3, [0, 1, 2, 2]*3, [.6]*12, .8, 400),
        ('aliased_groups', [2]*12, [0]*12, [.8]*6+[.4]*6, .8, 400),
        ('bounds_support', [1, 2, 3, 4, 2, 4, 1, 3], [0, 0, 0, 0, 2, 4, 1, 1],
         [.7, .8, .6, .5, 1., .4, 1., .4], 1., 400),
    ]
    inputs, evaluator = [], []
    for index, (family, major, minor, ccf, purity, depth) in enumerate(specs):
        major, minor, ccf = np.array(major), np.array(minor), np.array(ccf)
        multiplicity = np.ones(len(major), dtype=int)
        if family == 'aliased_groups':
            multiplicity[6:] = 2
        scale = purity/(2*(1-purity)+purity*(major+minor))
        alt = np.round(depth*scale*ccf*multiplicity).astype(int)
        if family == 'bounds_support':
            alt[0], alt[-1] = 0, depth
        labels = np.unique(ccf, return_inverse=True)[1]
        case, truth = write_case(root, f'mechanism-{index:02}', alt, depth-alt,
                                 major, minor, purity, labels, ccf, multiplicity)
        inputs.append(case)
        evaluator.append(dict(case, **truth, family=family))
    write_json(root/'FIT_MANIFEST.json', dict(schema='clipp1d.prior_panel.inputs.v1', cases=inputs,
                                             generator_sha256=sha(__file__)))
    write_json(root/'EVALUATION_MANIFEST.json', dict(cases=evaluator, purpose='mechanism, not accuracy acceptance',
                                                  generator_sha256=sha(__file__)))


def generate_performance(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root/'inputs').mkdir()
    (root/'truth').mkdir()
    rng = np.random.RandomState(2026092603)
    cases, truths = [], []
    for n in (256, 1000, 4000):
        for replicate in range(3):
            depth, purity, cna = (100, 200, 500)[replicate], (.4, .6, .9)[replicate], .4
            sizes = np.floor(n*np.array([.4, .25, .2, .15])).astype(int)
            sizes[0] += n-int(sizes.sum())
            labels = np.repeat(np.arange(4), sizes)
            rng.shuffle(labels)
            ccf = np.array([1., .75, .5, .25])[labels]
            major, minor = sample_copy_numbers(rng, n, cna)
            multiplicity = sample_multiplicity(rng, major)
            coverage = rng.poisson(depth, n)
            alt = rng.binomial(coverage, purity*ccf*multiplicity/(2*(1-purity)+purity*(major+minor)))
            case, truth = write_case(root, f'performance-{n}-{replicate}', alt, coverage-alt,
                                     major, minor, purity, labels, ccf, multiplicity)
            cases.append(case)
            truths.append(dict(case, **truth, depth=depth, purity=purity, cna_rate=cna))
    write_json(root/'FIT_MANIFEST.json', dict(schema='clipp1d.prior_panel.inputs.v1', cases=cases,
                                             generation_seed=2026092603, generator_sha256=sha(__file__)))
    write_json(root/'EVALUATION_MANIFEST.json', dict(cases=truths, purpose='cost fixtures only',
                                                  generator_sha256=sha(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--stage', choices=['development', 'confirmation', 'mechanisms', 'performance'], required=True)
    args = parser.parse_args()
    if args.stage == 'mechanisms':
        generate_mechanisms(args.outdir)
    elif args.stage == 'performance':
        generate_performance(args.outdir)
    else:
        print(generate_panel(args.outdir, args.stage))


if __name__ == '__main__':
    main()
