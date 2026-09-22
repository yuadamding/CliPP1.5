"""Hash-bind every true single-region case and normalize simulator truth.

Preparation is read-only for canonical datasets. SimClone repairs are separate
files. No inference, outcome-based selection, or region projection occurs here.
"""
import argparse
from collections import Counter
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from clipp1d.io import read_tumor
from clipp1d.report import write_json
from repair_simclone_normal_cn import repair

WORK = Path('/data/CliPP2')
SIM = Path('/data/CliPP_Sim')
FIELDS = ['mutation_id', 'true_cluster', 'true_ccf', 'true_multiplicity',
          'major_cn', 'minor_cn', 'mixed_cn']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader((line for line in stream if not line.startswith('#')), delimiter='\t'))


def coord(row, chromosome, position):
    value = Decimal(row[position])
    if value != value.to_integral_value():
        raise ValueError('Truth position is not an integer')
    return 'chr' + row[chromosome].removeprefix('chr') + ':' + str(int(value))


def unique(table, chromosome, position, fields):
    result = {}
    for row in table:
        key = coord(row, chromosome, position)
        value = tuple(row[f] for f in fields)
        if key in result and result[key] != value:
            result[key] = None  # Ambiguous coordinate: never invent its truth.
        else:
            result[key] = value
    return result


def inventory():
    cases = []
    root = WORK / 'CliPP2Sim1K_regionalCN_CNA0246_20260917'
    manifest = rows(root / 'cohort_manifest.tsv')
    for row in manifest:
        if int(row['region_count']) == 1:
            cases.append(dict(dataset=root.name, case_id=row['tumor_id'],
                              input_path=str(root / row['input_file']), metadata=row,
                              expected_sha256=row['input_sha256']))
    for dataset in ['SimClone1000_TSV', 'PhylogicNDT500_TSV']:
        for row in rows(WORK / dataset / 'conversion_manifest.tsv'):
            cases.append(dict(dataset=dataset, case_id=row['tumor_id'], input_path=row['output_file'],
                              metadata=row, expected_sha256=row['output_sha256']))
    design = {r['sample']: r for r in rows(WORK / 'CliPPSim4K_generated/generation_summary.tsv')}
    for path in sorted((WORK / 'CliPPSim4K_tsv').glob('*.tsv')):
        cases.append(dict(dataset='CliPPSim4K_tsv', case_id=path.stem, input_path=str(path),
                          metadata=design[path.stem], expected_sha256=sha(path)))
    assert Counter(c['dataset'] for c in cases) == {
        'CliPP2Sim1K_regionalCN_CNA0246_20260917': 200,
        'SimClone1000_TSV': 756, 'PhylogicNDT500_TSV': 500, 'CliPPSim4K_tsv': 4000}
    return cases


def prepare(case, root):
    directory = root / 'cases' / case['dataset'] / case['case_id']
    directory.mkdir(parents=True)
    source = Path(case['input_path'])
    assert sha(source) == case['expected_sha256'], source
    case = dict(case, original_input_path=str(source), original_input_sha256=sha(source))
    if case['dataset'] == 'SimClone1000_TSV':
        corrected = directory / 'input.normal-cn2.tsv'
        case['normal_cn_repair'] = repair(source, corrected)
        case['input_path'] = str(corrected)
    source = Path(case['input_path'])
    case['input_sha256'] = sha(source)
    data = read_tumor(source)
    inputs = rows(source)
    assert len({r['sample_id'] for r in inputs}) == 1
    assert len(inputs) == len(data.mutations), 'Expected one clonal CN state per mutation'
    assert all(float(r['cn_state_fraction']) == 1 for r in inputs)
    case.update(input_mutations=len(data.mutations), retained_mutations=len(data.retained),
                excluded_mutations=len(data.mutations)-len(data.retained), purity=data.purity,
                mean_depth=sum(m.alt+m.ref for m in data.mutations if m.observed)/len(data.mutations),
                exclusions=dict(Counter(m.exclusion for m in data.mutations if m.exclusion)),
                normal_cn_counts=dict(Counter(str(m.normal_cn) for m in data.mutations)))
    truth_paths = []
    by_id = {}
    name = case['case_id']
    if case['dataset'] == 'SimClone1000_TSV':
        t = SIM / 'testing_SimClone1000_truth' / name / 'simulated_0001'
        truth_paths = [t/'truth_tree/simulated_0001_mutation_assignments.txt',
                       t/'truth_tree/simulated_0001_subclonal_structure.txt',
                       t/'simulated_0001_0001/truth/simulated_0001_0001_multiplicity.txt']
        label = unique(rows(truth_paths[0]), 'chr', 'pos', ['cluster'])
        structure = {r['cluster']: float(r['simulated_0001_0001']) for r in rows(truth_paths[1])}
        dosage = unique(rows(truth_paths[2]), 'chr', 'pos', ['ccf', 'multiplicity'])
        for row in inputs:
            key = coord(row, 'chromosome', 'position')
            if label.get(key) is None or dosage.get(key) is None:
                continue
            cluster, = label[key]
            ccf, mult = dosage[key]
            assert abs(float(ccf)-structure[cluster]) < 1e-10
            by_id[row['mutation_id']] = (cluster, ccf, mult)
    elif case['dataset'] == 'PhylogicNDT500_TSV':
        t = SIM / 'training_PhylogicNDT500_simulations_12_16'
        truth_paths = [t/'Mafs_Real_Info'/f'{name}.truth_vals.maf.tsv',
                       t/'Mut_Assign'/f'{name}.mutation_assignments.txt']
        label = unique(rows(truth_paths[1]), 'chr', 'pos', ['cluster'])
        fields = ['mut_ccf','integer_mult','mult','t_alt_count','t_ref_count',
                  'maj_a','min_a','frac_s','nMaj2_A','nMin2_A']
        dosage = unique(rows(truth_paths[0]), 'Chromosome', 'Start_position', fields)
        mixed = mismatch = 0
        for row in inputs:
            key = coord(row, 'chromosome', 'position')
            if label.get(key) is None or dosage.get(key) is None:
                continue
            ccf, integer, mult, alt, ref, major, minor, fraction, major2, minor2 = dosage[key]
            assert int(alt) == int(row['alt_count']) and int(ref) == int(row['ref_count'])
            is_mixed = 0 < float(fraction) < 1 and (major,minor) != (major2,minor2)
            mixed += is_mixed
            mismatch += sorted([float(major),float(minor)]) != sorted([float(row['allele_a_cn']),float(row['allele_b_cn'])])
            # The integer dosage field is explicit simulator truth. Retain any
            # effective-dosage disagreement as model-mismatch metadata.
            by_id[row['mutation_id']] = (*label[key], ccf, integer)
        case['truth_mixed_cn_mutations'] = mixed
        case['truth_primary_cn_mismatches'] = mismatch
        case['truth_effective_vs_integer_dosage_differences'] = sum(float(v[1]) != float(v[2]) for v in dosage.values() if v is not None)
    elif case['dataset'] == 'CliPPSim4K_tsv':
        truth_paths = [WORK/'CliPPSim4K_generated'/name/'truth.txt']
        for row in rows(truth_paths[0]):
            mid = 'chr'+row['chromosome_index']+'_'+row['position']
            assert mid not in by_id
            by_id[mid] = (row['cluster_id'],row['ccf'],row['multiplicity'])
        assert set(by_id) == {r['mutation_id'] for r in inputs}
    else:
        truth_paths = [source.parent/'truth.txt',source.parent/'truth_mutation_sample.tsv',
                       source.parent/'scenario_manifest.json']
        scenario = json.loads(truth_paths[2].read_text())
        assert sha(truth_paths[2]) == case['metadata']['scenario_manifest_sha256']
        for path in truth_paths[:2]:
            assert sha(path) == scenario['output_files'][path.name]['sha256']
        label = {r['mutation_id']:r['cluster_id'] for r in rows(truth_paths[0])}
        for row in rows(truth_paths[1]):
            assert row['sample_id'] == '0'
            by_id[row['mutation_id']] = (label[row['mutation_id']],row['ccf'],row['multiplicity'])
        assert set(by_id) == {r['mutation_id'] for r in inputs}
    normalized = []
    retained = {m.mutation_id for m in data.retained}
    for row in inputs:
        if row['mutation_id'] not in retained or row['mutation_id'] not in by_id:
            continue
        cluster, ccf, multiplicity = by_id[row['mutation_id']]
        assert 0 <= float(ccf) <= 1+1e-12
        assert float(multiplicity) == int(float(multiplicity)) and float(multiplicity) >= 1
        normalized.append(dict(mutation_id=row['mutation_id'],true_cluster=cluster,true_ccf=ccf,
                               true_multiplicity=int(float(multiplicity)),major_cn=row['allele_a_cn'],
                               minor_cn=row['allele_b_cn'],mixed_cn='0'))
    target = directory/'truth.tsv'
    with target.open('x') as stream:
        writer=csv.DictWriter(stream,FIELDS,delimiter='\t',lineterminator='\n')
        writer.writeheader()
        writer.writerows(normalized)
    case.update(truth_path=str(target),truth_sha256=sha(target),
                truth_sources={str(p):sha(p) for p in truth_paths},
                truth_k_retained=len({r['true_cluster'] for r in normalized}),
                truth_unmatched_retained=sorted(retained-set(by_id)),
                truth_coverage=len(normalized)/len(retained) if retained else 0,
                cna_retained=sum((int(r['major_cn']),int(r['minor_cn'])) != (1,1) for r in normalized),
                status='prepared')
    write_json(directory/'preparation.json',case)
    return case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--recover',type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    cases=inventory()
    imported=[]
    if args.recover:
        previous=json.loads((args.recover/'prepared.json').read_text())
        imported=[r for r in previous if r['status']=='prepared']
        for r in imported:
            assert sha(r['input_path']) == r['input_sha256']
            assert sha(r['truth_path']) == r['truth_sha256']
            r.setdefault('truth_unmatched_retained',[])
            r.setdefault('truth_coverage',1.0)
        completed={(r['dataset'],r['case_id']) for r in imported}
        cases=[r for r in cases if (r['dataset'],r['case_id']) not in completed]
    write_json(args.out/'inventory.json',cases)
    def one(case):
        try:
            return prepare(case,args.out)
        except Exception as error:
            failed=dict(case,status='preparation_failure',error_type=type(error).__name__,message=str(error))
            write_json(args.out/'cases'/case['dataset']/case['case_id']/'preparation-failure.json',failed)
            return failed
    results=list(imported)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,result in enumerate(pool.map(one,cases),1):
            results.append(result)
            if result['status'] != 'prepared' or i%100==0:
                print(i,result['dataset'],result['case_id'],result['status'],result.get('message',''),flush=True)
    write_json(args.out/'prepared.json',results)
    summary=dict(total=len(results),counts=dict(Counter(r['dataset'] for r in results)),
                 status=dict(Counter(r['status'] for r in results)))
    write_json(args.out/'preparation-summary.json',summary)
    print(summary,flush=True)


if __name__=='__main__':
    main()
