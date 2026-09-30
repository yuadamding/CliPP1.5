"""Source-bound allocated-CUDA tests and fresh baseline/candidate full-fit pairs.

The plan supplies inputs and immutable source inventories, never saved labels or
solver states. This worker submits no jobs and changes no installed environment.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

SCHEMA = "clipp1d.repository_improvement_qualification.v1"
QUALIFICATION_SCOPE = "execution_certificates_and_shared_model_contract"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def owned(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Payload paths must be relative and confined")
    path = root / relative
    if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
        raise ValueError("Payload paths must not traverse symlinks")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Payload path escaped root")
    return path


def source_inventory(root):
    package = root / "src/clipp1d"
    files = {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob("*.py"))}
    if not files or any(p.is_symlink() for p in package.rglob("*")):
        raise ValueError("Empty or linked source package")
    digest = hashlib.sha256()
    for name, value in files.items():
        digest.update(f"{name}\0{value}\n".encode())
    return dict(source_sha256=digest.hexdigest(), source_files=files)


def validate_plan(root, plan, mode):
    if plan.get("schema") != "clipp1d.repository_improvement_plan.v1":
        raise ValueError("Wrong repository qualification plan")
    if plan.get("saved_partition_or_solver_state_inputs") is not False:
        raise ValueError("This qualification must start fresh from input files")
    if plan.get("worker_sha256") != sha(__file__):
        raise ValueError("Worker identity differs")
    for label in ("baseline", "candidate"):
        item = plan[label]
        actual = source_inventory(owned(root, item["root"]))
        if actual != {k: item[k] for k in ("source_sha256", "source_files")}:
            raise ValueError(f"{label} source inventory changed")
    for name, digest in plan["assets"].items():
        path = owned(root, name)
        if not path.is_file() or sha(path) != digest:
            raise ValueError("Required qualification asset differs: " + name)
    config = plan["configuration"]
    if config.get("max_major_cn") != 4 or config.get("partition_search") is not True:
        raise ValueError("Bind the original input/model policy and separate partition estimator")
    if config.get("generic_partition_grouping") is not False:
        raise ValueError("Grouping ablations are outside this shared comparison")
    if config.get('baseline_partition_policy') != {} or config.get('candidate_partition_policy') != {'refit_relocations': True}:
        raise ValueError("Only the explicitly enabled refit-relocation phase may differ in partition policy")
    for name in ("tests", "fit"):
        limit = plan["timeouts"][name]
        if type(limit) is not int or not 1 <= limit <= 172800:
            raise ValueError("Finite explicit execution limits required")
    inputs = plan["synthetic_inputs"] if mode == "tests" else [plan["input"]]
    if not inputs:
        raise ValueError("At least one fresh input is required")
    for item in inputs:
        if item["sha256"] != sha(owned(root, item["path"])):
            raise ValueError("Fresh input hash differs")
    if mode == "tests":
        test = plan["tests"]
        vendor = owned(root, test["vendor_root"])
        manifest_path = owned(root, test["vendor_manifest"])
        if sha(manifest_path) != test["vendor_manifest_sha256"]:
            raise ValueError("Test vendor authority differs")
        manifest = json.loads(manifest_path.read_text())
        actual = {str(p.relative_to(vendor)): sha(p) for p in vendor.rglob("*") if p.is_file()}
        if actual != manifest["files"] or any(p.is_symlink() for p in vendor.rglob("*")):
            raise ValueError("Frozen pytest vendor inventory differs")
        if not test["paths"] or len(test["paths"]) != len(set(test["paths"])):
            raise ValueError("Unique frozen test file paths required")
        for name in test["paths"]:
            if not name.startswith("tests/") or not name.endswith(".py"):
                raise ValueError("Explicit test files required")
            if str(Path(plan['candidate']['root']) / name) not in plan["assets"]:
                raise ValueError("Unbound test file")
        cuda_files = test['allocated_cuda_files']
        if (not cuda_files or not set(cuda_files).issubset(test['paths'])
                or not test['allocated_cuda_required_nodes']):
            raise ValueError("Explicit allocated-CUDA oracle scope required")
        if set(cuda_files) & set(test['cpu_reference_files']) or set(cuda_files) | set(test['cpu_reference_files']) != set(test['paths']):
            raise ValueError("CUDA versus CPU-reference test scope must reconcile")


def validate_test_report(report, declared_files, required_nodes=()):
    collected, passed = report["collected"], report["passed_calls"]
    if not collected or len(set(collected)) != len(collected):
        raise ValueError("Empty or duplicate collected tests")
    if Counter(collected) != Counter(passed) or report["nonpassing"]:
        raise ValueError("All collected tests must pass with zero skips")
    if {n.split("::")[0] for n in collected} != set(declared_files):
        raise ValueError("Declared test files missing from collection")
    if not set(required_nodes).issubset(collected):
        raise ValueError("Required CUDA or regression oracle missing")
    return dict(tests=len(collected), failures=0, errors=0, skipped=0,
                collected_node_ids=collected, required_node_ids=list(required_nodes))


def run_child(root, plan_path, expected, out, phase, label="candidate", input_item=None, baseline=None):
    plan = json.loads(plan_path.read_text())
    source = owned(root, plan[label]["root"])
    paths = [str(source / "src"), str(source)]
    if phase == "tests":
        paths.insert(0, str(owned(root, plan["tests"]["vendor_root"])))
    bootstrap = "import json,runpy,sys; sys.dont_write_bytecode=True; sys.path[:0]=json.loads(sys.argv[1]); sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    command = [sys.executable, "-I", "-B", "-c", bootstrap, json.dumps(paths), str(Path(__file__).resolve()),
        "--child", phase, "--label", label, "--payload-root", str(root), "--plan", str(plan_path),
        "--expected-plan-sha256", expected, "--expected-source-sha256", plan['candidate']['source_sha256'],
        "--out", str(out)]
    if input_item is not None:
        command += ["--input-json", json.dumps(input_item, sort_keys=True)]
    if baseline is not None:
        command += ["--baseline-observation", str(baseline)]
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="", PYTEST_PLUGINS="")
    env.pop("PYTHONPATH", None)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "child.log").open("x") as log:
        try:
            result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=plan["timeouts"][phase])
            status = dict(returncode=result.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            status = dict(returncode=None, timed_out=True)
            write_json(out/'subprocess-status.json', status)
            raise TimeoutError("Bounded child timed out; abort all further phases and retain outer cleanup authority")
    write_json(out/'subprocess-status.json', status)
    report = out / "observation.json"
    status["observation"] = json.loads(report.read_text()) if report.is_file() else None
    return status


def gpu_identity(source, expected):
    import torch
    import clipp1d
    from clipp1d.api import source_provenance
    if Path(clipp1d.__file__).resolve().parent != source / "src/clipp1d":
        raise ValueError("Wrong imported source package")
    if source_provenance()["source_sha256"] != expected:
        raise ValueError("Imported source hash differs")
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name(0)):
        raise ValueError("Exactly one allocated A100 required")
    torch.set_num_threads(1)
    return dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                interpreter=sys.executable, source_sha256=expected)


def pytest_runtime_directory(root, out):
    """Keep pytest's temporary links in the exact task's existing runtime tree."""
    run_root = root.parent
    raw_tmp = os.environ.get('TMPDIR', '')
    tmp = Path(raw_tmp)
    if not raw_tmp or not tmp.is_absolute():
        raise ValueError("An absolute worker-owned TMPDIR is required")
    try:
        relative = tmp.relative_to(run_root / 'runtime')
    except ValueError:
        raise ValueError("Pytest TMPDIR must be inside this worker's runtime tree") from None
    if len(relative.parts) != 2 or relative.parts[1] != 'tmp':
        raise ValueError("Pytest TMPDIR must match runtime/<task-key>/tmp")
    key = relative.parts[0]
    tmp = owned(run_root, Path('runtime') / key / 'tmp')
    try:
        output_relative = out.relative_to(run_root / 'outputs' / key)
    except ValueError:
        raise ValueError("Pytest output and TMPDIR task identities differ") from None
    output = owned(run_root, Path('outputs') / key / output_relative)
    for directory in (tmp, tmp.parent, output):
        if not directory.is_dir() or directory.stat().st_uid != os.getuid():
            raise ValueError("Pytest runtime and output directories must exist and belong to this worker")
    # Pytest removes a pre-existing --basetemp. Give it a fresh absent child of
    # our exclusively created container, preserving all prior runtime evidence.
    container = Path(tempfile.mkdtemp(prefix='repository-pytest-', dir=tmp))
    basetemp = owned(run_root, container.relative_to(run_root) / 'basetemp')
    write_json(out / 'pytest-runtime.json', dict(
        schema='clipp1d.repository_pytest_runtime.v1', task_key=key,
        worker_tmpdir=str(tmp), basetemp=str(basetemp),
        outside_scientific_outputs=True, retained_for_diagnosis=True))
    return basetemp


def child_tests(root, plan, out):
    from importlib import import_module, metadata
    import pytest
    source = owned(root, plan['candidate']['root'])
    vendor = owned(root, plan['tests']['vendor_root'])
    if not Path(pytest.__file__).resolve().is_relative_to(vendor):
        raise ValueError("Pytest was not imported from frozen vendor")
    manifest = json.loads(owned(root, plan['tests']['vendor_manifest']).read_text())
    dependency_imports = {}
    for name in ('osqp', 'joblib'):
        module = import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(vendor):
            raise ValueError("Test-only dependency imported outside frozen vendor: " + name)
        expected = manifest['distributions'][name]['version']
        if metadata.version(name) != expected:
            raise ValueError("Test-only dependency version differs: " + name)
        dependency_imports[name] = dict(version=expected, imported_from_frozen_vendor=True)
    extension = import_module('osqp.ext_builtin')
    if not Path(extension.__file__).resolve().is_relative_to(vendor):
        raise ValueError("OSQP extension ABI/import is not from frozen test vendor")
    hardware = gpu_identity(source, plan['candidate']['source_sha256'])
    basetemp = pytest_runtime_directory(root, out)
    evidence = dict(collected=[], passed_calls=[], nonpassing=[])

    class Collector:
        def pytest_collection_modifyitems(self, items):
            evidence['collected'] = [item.nodeid for item in items]
            write_json(out / "collection.json", dict(node_ids=evidence['collected']))

        def pytest_runtest_logreport(self, report):
            if report.when == 'call' and report.passed:
                evidence['passed_calls'].append(report.nodeid)
            if not report.passed:
                evidence['nonpassing'].append(dict(nodeid=report.nodeid, when=report.when,
                                                   outcome=report.outcome))

    try:
        code = pytest.main(["-q", *plan['tests']['paths'], "--rootdir="+str(source),
            "-c", str(source / 'pyproject.toml'), "-p", "no:cacheprovider",
            "--basetemp="+str(basetemp), "--junitxml="+str(out/'pytest.xml')], plugins=[Collector()])
    finally:
        write_json(out / "test-results.json", evidence)
    required = plan['tests']['allocated_cuda_required_nodes']
    summary = validate_test_report(evidence, plan['tests']['paths'], required)
    if code:
        raise ValueError("Allocated test suite failed")
    return dict(status="completed", qualification_passed=True, hardware=hardware, **summary,
        test_only_dependency_imports=dependency_imports, osqp_extension_import_passed=True,
        pytest_runtime_receipt='pytest-runtime.json',
        allocated_cuda_files=plan['tests']['allocated_cuda_files'],
        allocated_cuda_required_nodes=required, cpu_reference_files=plan['tests']['cpu_reference_files'],
        cpu_reference_tests_are_not_cuda_qualification=True)


def child_fit(root, plan, out, label, item, baseline_path):
    import numpy as np
    import torch
    from clipp1d import cuda_api
    from clipp1d.cuda.refinement import PartitionSearchPolicy

    source = owned(root, plan[label]['root'])
    hardware = gpu_identity(source, plan[label]['source_sha256'])
    input_path = owned(root, item['path'])
    allowed_inputs = plan['synthetic_inputs'] if plan['mode'] == 'tests' else [plan['input']]
    if item not in allowed_inputs:
        raise ValueError("Child fit input is not an exact planned descriptor")
    if sha(input_path) != item['sha256']:
        raise ValueError("Input changed before fresh fit")
    baseline = json.loads(baseline_path.read_text()) if baseline_path is not None else None
    config = plan['configuration']
    policy = PartitionSearchPolicy(**config[label+'_partition_policy'])
    original = cuda_api.fit_tensor_model
    observation, calls = {}, 0

    def observe(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        graph = result.graph
        if label == 'baseline':
            path = out / 'graph-weights.npy'
            weights = np.lib.format.open_memmap(path, mode='w+', dtype='<f8', shape=(graph.n, graph.n))
            for first in range(0, graph.n, 128):
                weights[first:first+128] = graph.weights[first:first+128].detach().cpu().numpy()
            weights.flush()
            del weights
            observation['graph_weights'] = dict(path=path.name, sha256=sha(path), n=graph.n)
        elif baseline and baseline.get('graph_weights') and baseline.get('mutation_ids') == list(graph.mutation_ids):
            descriptor = baseline['graph_weights']
            path = baseline_path.parent / descriptor['path']
            if path.name != 'graph-weights.npy' or path.is_symlink() or sha(path) != descriptor['sha256']:
                raise ValueError("Baseline graph observation differs")
            weights = np.load(path, mmap_mode='r', allow_pickle=False)
            if weights.shape != (graph.n, graph.n) or weights.dtype != np.float64:
                raise ValueError("Baseline graph observation shape/dtype differs")
            difference = 0.
            for first in range(0, graph.n, 128):
                actual = graph.weights[first:first+128].detach().cpu().numpy()
                difference = max(difference, float(np.max(np.abs(actual-weights[first:first+128]))))
            observation['graph_max_abs_weight_difference'] = difference
        observation.update(graph_rule=graph.weight_rule, graph_edges=graph.edges)
        graph.validate(full=True)
        return result

    cuda_api.fit_tensor_model = observe
    try:
        result = cuda_api.fit(input_path, out/'fit', max_major_cn=config['max_major_cn'],
            device='cuda:0', partition_search=True, generic_partition_grouping=False, partition_policy=policy)
    finally:
        cuda_api.fit_tensor_model = original
    if calls != 1:
        raise ValueError("Full-fit observer did not surround exactly one original invocation")
    arrays = {name: getattr(result, name) for name in ('pilot_phi', 'raw_phi', 'refitted_phi',
        'cluster_labels', 'partition_labels', 'original_lower_bounds', 'original_upper_bounds',
        'multiplicity_calls', 'refitted_multiplicity_calls')}
    np.savez_compressed(out/'arrays.npz', **arrays)
    candidate = result.candidate_provenance
    if candidate['raw_qualified'] is not True or candidate['refit_qualified'] is not True:
        raise ValueError("Published primary raw/refit candidate is unqualified")
    partition = result.partition_estimate
    observation.update(status='completed', qualification_passed=True, hardware=hardware,
        source_sha256=plan[label]['source_sha256'], input_sha256=item['sha256'],
        model_sha256=result.provenance['model_sha256'], mutation_ids=list(result.mutation_ids),
        graph_sha256=result.graph_sha256, graph_recipe=result.provenance['graph'],
        search_status=result.search_status, raw_diagnostics=result.raw_diagnostics,
        raw_objective=result.raw_objective, selection_score=result.selection_score,
        selected_lambda=result.selected_lambda, source_provenance=result.provenance,
        candidate_provenance=candidate, operation_metrics=result.operation_metrics,
        arrays=dict(path='arrays.npz', sha256=sha(out/'arrays.npz')),
        partition=None if partition is None else dict(score=partition.score, status=partition.search_status,
            provenance=partition.provenance, cluster_sizes=np.bincount(partition.memberships).tolist(),
            membership_sha256=hashlib.sha256(partition.memberships.astype('<i8').tobytes()).hexdigest()),
        selected_candidate_qualified=True, full_raw_search_complete=result.search_status=='complete',
        scalar_or_qp_state_imported=False)
    run = json.loads((out/'fit/run.json').read_text())
    expected_schema = 'clipp1d.cuda.run.v5' if label == 'baseline' else 'clipp1d.cuda.run.v6'
    if run.get('schema') != expected_schema or run.get('status') != 'success':
        raise ValueError("Fresh public fit schema/status differs from the bound estimator scope")
    observation['public_schema'] = run['schema']
    if source_inventory(source)['source_sha256'] != plan[label]['source_sha256']:
        raise ValueError("Source changed during fit")
    torch.cuda.synchronize()
    return observation


def compare_pair(baseline, candidate, before, after):
    """Comparison arithmetic executes only in the allocated worker process."""
    b, c = baseline.get('observation'), candidate.get('observation')
    qualified = all(valid_child(x) for x in (baseline, candidate))
    result = dict(baseline=baseline, candidate=candidate, paired_fits_qualified=qualified,
                  selected_outputs_promoted=False, held_out_biological_accuracy_qualified=False,
                  qualification_scope=QUALIFICATION_SCOPE, improvement_adoption_qualified=False)
    if not qualified:
        return result
    import numpy as np
    same_input_model = b['input_sha256'] == c['input_sha256'] and b['model_sha256'] == c['model_sha256']
    same_rows = b['mutation_ids'] == c['mutation_ids']
    same_policy = b['source_provenance']['policy'] == c['source_provenance']['policy']
    recipe = ('rule', 'representation', 'edge_count', 'mutation_ids')
    same_recipe = all(b['graph_recipe'][key] == c['graph_recipe'][key] for key in recipe)
    result.update(same_input_model=same_input_model, same_retained_mutations=same_rows,
        same_original_numerical_policy=same_policy,
        same_graph_construction_rule=same_recipe, graph_bytes_equal=b['graph_sha256']==c['graph_sha256'],
        graph_max_abs_weight_difference=c.get('graph_max_abs_weight_difference'),
        raw_objective_difference=c['raw_objective']-b['raw_objective'],
        raw_objective_comparable=b['graph_sha256']==c['graph_sha256'] and b['selected_lambda']==c['selected_lambda'],
        selection_score_difference=c['selection_score']-b['selection_score'],
        partition_score_difference=c['partition']['score']-b['partition']['score'],
        all_six_or_full_cohort_improved=False)
    if same_rows:
        for folder, report in ((before, b), (after, c)):
            if sha(folder/report['arrays']['path']) != report['arrays']['sha256']:
                raise ValueError("Paired output array observation differs")
        with np.load(before/b['arrays']['path'], allow_pickle=False) as ba, np.load(after/c['arrays']['path'], allow_pickle=False) as ca:
            result['arrays'] = {name: dict(exactly_equal=bool(np.array_equal(ba[name], ca[name])),
                max_abs_difference=float(np.max(np.abs(ba[name]-ca[name])))) for name in ba.files}
    result['qualification_passed'] = qualified and same_input_model and same_rows and same_recipe and same_policy
    return result


def valid_child(status):
    return (status.get('returncode') == 0 and status.get('timed_out') is False
            and (status.get('observation') or {}).get('qualification_passed') is True)


def pair(root, plan_path, expected, out, item):
    out.mkdir(parents=True, exist_ok=False)
    before, after = out/'baseline', out/'candidate'
    b = run_child(root, plan_path, expected, before, 'fit', 'baseline', item)
    previous = before/'observation.json' if (before/'observation.json').is_file() else None
    c = run_child(root, plan_path, expected, after, 'fit', 'candidate', item, previous)
    return compare_pair(b, c, before, after)


def artifacts(out):
    return [dict(path=str(p.relative_to(out)), sha256=sha(p), size=p.stat().st_size)
            for p in sorted(out.rglob('*')) if p.is_file() and not p.is_symlink()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('tests', 'paired'))
    parser.add_argument('--child', choices=('tests', 'fit'))
    parser.add_argument('--label', choices=('baseline', 'candidate'), default='candidate')
    parser.add_argument('--input-json')
    parser.add_argument('--baseline-observation', type=Path)
    for option in ('payload-root', 'plan', 'out'):
        parser.add_argument('--'+option, type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    report = dict(schema=SCHEMA, status='failed', qualification_passed=False,
                  worker_sha256=sha(__file__), mode=args.child or args.mode,
                  qualification_scope=QUALIFICATION_SCOPE, improvement_adoption_qualified=False)
    child = args.child is not None
    if not child:
        args.out.mkdir(parents=True, exist_ok=False)
    try:
        if sha(args.plan) != args.expected_plan_sha256:
            raise ValueError("Plan hash differs")
        plan = json.loads(args.plan.read_text())
        root = args.payload_root.resolve()
        if plan['candidate']['source_sha256'] != args.expected_source_sha256:
            raise ValueError("Candidate source literal differs")
        if (not child and args.mode != plan['mode']) or plan['mode'] not in ('tests', 'paired'):
            raise ValueError("Invocation mode differs from frozen plan")
        validate_plan(root, plan, plan['mode'])
        if child:
            if args.child == 'tests':
                report.update(child_tests(root, plan, args.out))
            else:
                report.update(child_fit(root, plan, args.out, args.label,
                                        json.loads(args.input_json), args.baseline_observation))
        else:
            if args.mode == 'tests':
                tests = run_child(root, args.plan, args.expected_plan_sha256, args.out/'tests', 'tests')
                report['tests'] = tests
                if not valid_child(tests):
                    raise ValueError("Shared allocated test gate failed; no synthetic fits executed")
                pairs = [pair(root, args.plan, args.expected_plan_sha256, args.out/f'synthetic-{i:02d}', item)
                         for i, item in enumerate(plan['synthetic_inputs'])]
            elif args.mode == 'paired':
                pairs = [pair(root, args.plan, args.expected_plan_sha256, args.out/'pair', plan['input'])]
            else:
                raise ValueError("An explicit mode is required")
            report.update(status='completed', pairs=pairs,
                qualification_passed=all(p.get('qualification_passed') is True for p in pairs),
                fresh_repository_fullfit_scope=True, saved_partition_or_solver_state_inputs=False,
                source_commit=plan.get('source_commit'), baseline_source_sha256=plan['baseline']['source_sha256'],
                candidate_source_sha256=plan['candidate']['source_sha256'],
                plan_sha256=args.expected_plan_sha256, configuration=plan['configuration'])
            validate_plan(root, plan, args.mode)
    except BaseException as error:
        report.update(status='failed', qualification_passed=False, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
    report.update(seconds=time.monotonic()-started, utc=datetime.now(timezone.utc).isoformat())
    if child and args.child == 'tests':
        evidence_path = args.out / 'test-results.json'
        if evidence_path.is_file() and not evidence_path.is_symlink():
            report['test_results'] = json.loads(evidence_path.read_text())
    if not child:
        report['artifacts'] = artifacts(args.out)
    write_json(args.out/('observation.json' if child else 'receipt.json'), report)
    return 0 if report['qualification_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
