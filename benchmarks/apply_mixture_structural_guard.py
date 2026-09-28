"""Apply a frozen structural selector to a complete saved development inventory."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import pandas as pd
import torch

from run_mixture_experiment import MixturePolicy, prepare_model, sha, write_json
from mixture_structural_guard import POLICY, choose


def read(path):
    return json.loads(Path(path).read_text())


def preserved_metadata(parent, baseline):
    """An unchanged hard partition has no fitted soft-mixture masses/posterior.

    Keep its actual centers, rather than accidentally attaching the rejected
    soft candidate's centers or latent masses to the preserved table.
    """
    centers = baseline.groupby('cluster_label').partition_ccf.first().sort_index()
    return dict(parent, score=None, log_likelihood=None, adaptive=False,
                centers=centers.tolist(), baseline_label_order=centers.index.tolist(),
                weights=None, enrichment=None, shared_feasible_interval=None,
                status='preserved_baseline', selected_parameters_converged=None,
                search_status='original_partition_status_not_reinterpreted',
                ccf_estimate='preserved_original_complete_graph_partition_refit')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--result-dirs', nargs='+', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-reference', action='store_true')
    args = parser.parse_args(argv)
    if args.device.startswith('cpu') != args.cpu_reference:
        parser.error('CPU selection requires --device cpu and --cpu-reference')
    manifest = read(args.manifest)
    if manifest.get('schema') != 'clipp1d.mixture_experiment_inputs.v1':
        raise ValueError('Unrecognized inference manifest')
    policy = MixturePolicy(**manifest['policy'])
    cases = {c['case_id']: c for c in manifest['cases']}
    if not cases or len(cases) != len(manifest['cases']):
        raise ValueError('Expected a nonempty unique case inventory')
    parents, authorities, numerical_inventory = {}, {}, None
    for directory in map(Path, args.result_dirs):
        binding, terminal = read(directory/'BINDING.json'), read(directory/'COMPLETE.json')
        if not terminal['source_unchanged'] or not terminal['manifest_unchanged']:
            raise ValueError('Changed parent source or manifest')
        numerical = {k: v for k, v in binding['source_inventory'].items() if k.startswith('src/')}
        if numerical_inventory is None:
            numerical_inventory = numerical
        if numerical != numerical_inventory or binding['policy'] != asdict(policy):
            raise ValueError('Different parent numerical source or policy')
        authorities[str(directory)] = dict(binding_sha256=sha(directory/'BINDING.json'),
                                            terminal_sha256=sha(directory/'COMPLETE.json'),
                                            failures=terminal['failures'])
        for r in terminal['results']:
            case_id = r['case_id']
            if case_id in parents:
                raise ValueError('Duplicate successful parent case')
            path = Path(r['output'])/'EXPERIMENT.json'
            parents[case_id] = path
    if set(parents) != set(cases):
        raise ValueError('Missing or unexpected development cases')
    repo = Path(__file__).resolve().parents[1]
    for name, digest in numerical_inventory.items():
        if sha(repo/name) != digest:
            raise ValueError('Selector model does not match parent numerical source')
    inventory = dict(numerical_inventory)
    for name in ('run_mixture_experiment.py', 'mixture_structural_guard.py', 'apply_mixture_structural_guard.py'):
        inventory['benchmarks/'+name] = sha(repo/'benchmarks'/name)
    root = Path(args.outdir).resolve()
    root.mkdir()
    write_json(root/'BINDING.json', dict(schema='clipp1d.structural_mixture_binding.v1',
        source_inventory=inventory, policy=asdict(policy), selection_policy=POLICY,
        manifest_sha256=sha(args.manifest), parent_authorities=authorities,
        python=sys.executable, torch=torch.__version__, selector_device=args.device,
        selector_execution='cpu_component_reference' if args.cpu_reference else 'compiled_cuda',
        production_adopted=False))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    outcomes = []
    for index, (case_id, case) in enumerate(cases.items()):
        started = time.perf_counter()
        parent_path = parents[case_id]
        parent = read(parent_path)
        if parent['inputs'] != case or parent['policy'] != asdict(policy):
            raise ValueError('Different parent data or scientific settings')
        for filename, digest in parent['output_sha256'].items():
            if sha(parent_path.parent/filename) != digest:
                raise ValueError('Changed parent output')
        model, _ = prepare_model(case, args.device)
        baseline = pd.read_csv(case['seed_path'], sep='\t', dtype={'mutation_id': str},
                               float_precision='round_trip').set_index('mutation_id').loc[list(model.mutation_ids)]
        result, decision = choose(model, parent['candidates'], policy,
                                   baseline_k=int(baseline.cluster_label.nunique()))
        if result is None:
            table = baseline[['cluster_label', 'partition_ccf', 'multiplicity_call']].rename(
                columns={'partition_ccf': 'mixture_ccf', 'multiplicity_call': 'multiplicity'})
            table = table.reset_index()
            metadata = preserved_metadata(parent, baseline)
        else:
            public = result.public_arrays(model.mutation_ids)
            table = pd.DataFrame(dict(mutation_id=model.mutation_ids,
                cluster_label=public['labels'].cpu().numpy(), mixture_ccf=public['ccf'].cpu().numpy(),
                multiplicity=public['multiplicity'].cpu().numpy()))
            metadata = dict(result.metadata(), centers=result.centers.tolist(),
                             weights=result.weights.tolist(), enrichment=result.enrichment.tolist())
        destination = root/f'{index:05}'
        destination.mkdir()
        table.to_csv(destination/'mixture_mutation_clusters.tsv', sep='\t', index=False, float_format='%.17g')
        centers = table.groupby('cluster_label').mixture_ccf.agg(['first', 'size']).reset_index().rename(
            columns={'first': 'mixture_ccf', 'size': 'n_mutations'})
        centers.to_csv(destination/'mixture_cluster_centers.tsv', sep='\t', index=False, float_format='%.17g')
        metadata.update(schema='clipp1d.experimental_structural_mixture.v1', case_id=case_id,
            inputs=case, policy=asdict(policy), selection_policy=POLICY, structural_decision=decision,
            selector_device=args.device,
            selector_execution='cpu_component_reference' if args.cpu_reference else 'compiled_cuda',
            elapsed_seconds=parent['elapsed_seconds']+time.perf_counter()-started,
            parent_receipt=str(parent_path), parent_receipt_sha256=sha(parent_path),
            source='same numerical model plus separately bound structural selector',
            original_baseline_preserved=result is None,
            output_sha256={name: sha(destination/name) for name in
                            ('mixture_mutation_clusters.tsv', 'mixture_cluster_centers.tsv')})
        write_json(destination/'EXPERIMENT.json', metadata)
        outcomes.append(dict(case_id=case_id, output=str(destination), selected_status=metadata['status']))
        print(json.dumps(dict(case_id=case_id, family=decision['selected_family'])), flush=True)
    unchanged = all(sha(repo/name) == digest for name, digest in inventory.items())
    write_json(root/'COMPLETE.json', dict(results=outcomes, failures=[], source_unchanged=unchanged,
        manifest_unchanged=sha(args.manifest) == read(root/'BINDING.json')['manifest_sha256'],
        source_scope='complete frozen component-study inventory', accuracy_acceptance=False,
        numerical_qualification=False))
    return 0 if unchanged else 1


if __name__ == '__main__':
    raise SystemExit(main())
