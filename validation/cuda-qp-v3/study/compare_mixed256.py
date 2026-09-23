"""Compare retained E/current and G/baseline evidence; no numerical fitting.

Prepare only: invoke after the parent confirms G's terminal import. Never alters
the originals. Writes one new G/DIAGNOSIS.json, refusing an existing destination.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


BASE = Path(__file__).resolve().parent
IDENTITIES = {
    "current": (
        "mixed256-e",
        "mixed256-current",
        "726eaf873550d80cd89d72d7c1253e1c70aa97135011fbad7f9d6c97c9382a88",
    ),
    "baseline": (
        "baseline256-g",
        "mixed256-baseline",
        "ea7071788094e0cd3537530fcdc62e4228054fce19ff02ba8f2cf4eacce1dd26",
    ),
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing regular evidence: {path}")
    require(all(not part.is_symlink() for part in path.parents), f"Symlink ancestry: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(payload):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON: {value}")

    return json.loads(payload, parse_constant=invalid)


def load(path):
    sha(path)
    return parse(path.read_bytes())


def safe(root, name):
    path = Path(name)
    require(
        not path.is_absolute() and path.parts and ".." not in path.parts, "Unsafe artifact path"
    )
    return root / path


def read_attempt(role):
    attempt_name, task_name, source_sha = IDENTITIES[role]
    parent = BASE / attempt_name
    imported = parent / "imported"
    manifest = load(parent / "IMPORT_MANIFEST.json")
    require(manifest["files"], "Empty terminal import")
    bindings = {}
    for name, item in manifest["files"].items():
        path = safe(imported, name)
        require(
            sha(path) == item["sha256"] and path.stat().st_size == item["bytes"],
            f"Imported evidence changed: {path}",
        )

    def bound(name):
        require(name in manifest["files"], f"Artifact not covered by import manifest: {name}")
        path = safe(imported, name)
        bindings[str(path.relative_to(BASE))] = dict(sha256=sha(path), bytes=path.stat().st_size)
        return load(path)

    plan = load(parent / "PREPARED.json")
    accepted, terminal = bound("receipts/accepted.json"), bound("receipts/terminal.json")
    sealed_plan = parent / "sealed" / plan["run_id"] / "PREPARED.json"
    require(
        terminal == manifest["terminal"]
        and terminal["job_id"] == accepted["job_id"]
        and accepted["job_name"] == plan["job_name"]
        and accepted["plan_sha256"] == terminal["plan_sha256"] == sha(sealed_plan),
        "Job/source-plan/terminal identity differs",
    )
    bindings[str(sealed_plan.relative_to(BASE))] = dict(
        sha256=sha(sealed_plan), bytes=sealed_plan.stat().st_size
    )
    receipt_name = f"results/{task_name}.json"
    receipt = bound(receipt_name)
    executed = [row for row in terminal["tasks"] if row["name"] == task_name]
    require(
        len(executed) == 1
        and executed[0]["receipt_sha256"] == manifest["files"][receipt_name]["sha256"]
        and executed[0]["status"] == receipt["status"],
        "Task receipt differs from worker terminal",
    )
    require(
        receipt["status"] in ("failed", "passed")
        and receipt["source"]["source_sha256"] == source_sha
        and plan["source_sha256"][role] == source_sha
        and receipt["nodes"] == 256
        and receipt["fixture_family"] == "mixed_support",
        "Unexpected source, fixture or task status",
    )
    require(
        receipt["cuda_available"] is True
        and receipt["device"].startswith("cuda:")
        and receipt["lsf_job_id"] == accepted["job_id"],
        "Task lacks matching allocated CUDA identity",
    )
    for name, digest in receipt["artifacts"].items():
        relative = "results/" + name
        require(
            relative in manifest["files"] and manifest["files"][relative]["sha256"] == digest,
            f"Task artifact differs from imported bytes: {relative}",
        )
    prefix = f"results/{task_name}.artifacts/"
    path = bound(prefix + "full-path.json")
    identity = bound(prefix + "input-identity.json")
    initial_plan = bound(prefix + "initial-path-plan.json")
    input_path = safe(imported, prefix + "input.tsv")
    require(sha(input_path) == receipt["input_sha256"], "Actual input bytes differ from receipt")
    bindings[str(input_path.relative_to(BASE))] = dict(
        sha256=sha(input_path), bytes=input_path.stat().st_size
    )
    events_path = safe(imported, "results/" + receipt["events_file"])
    require(sha(events_path) == receipt["events_sha256"], "Event journal hash differs")
    bindings[str(events_path.relative_to(BASE))] = dict(
        sha256=sha(events_path), bytes=events_path.stat().st_size
    )
    events = [parse(line) for line in events_path.read_bytes().splitlines()]
    candidates = {}
    for event in events:
        if event["kind"].startswith("candidate_"):
            rows = candidates.setdefault(event["index"], {})
            require(event["kind"] not in rows, "Duplicated candidate event")
            rows[event["kind"]] = event
    require(
        len(path["path_records"]) == len(initial_plan["initial_candidates"]) == 26,
        "Incomplete or changed literal initial path inventory",
    )
    for index, row in enumerate(path["path_records"]):
        planned = initial_plan["initial_candidates"][index]
        require(
            row["lambda_value"] == planned["lambda_value"],
            "Returned path differs from its sealed plan",
        )
        require(
            index in candidates and "candidate_started" in candidates[index],
            "Missing attempted candidate event",
        )
        require(
            all(
                event["lambda_value"] == row["lambda_value"] for event in candidates[index].values()
            ),
            "Candidate event penalty mismatch",
        )
    return dict(
        role=role,
        attempt=attempt_name,
        job_id=accepted["job_id"],
        receipt=receipt,
        path=path,
        initial_plan=initial_plan,
        input_identity=identity,
        candidates=candidates,
        source_artifacts=bindings,
    )


def flattened(value):
    if isinstance(value, list):
        return [number for child in value for number in flattened(child)]
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        "Comparison value is not finite numeric data",
    )
    return [float(value)]


def shape(value):
    if not isinstance(value, list):
        return ()
    shapes = [shape(child) for child in value]
    require(not shapes or all(s == shapes[0] for s in shapes), "Ragged scientific array")
    return (len(value),) + (shapes[0] if shapes else ())


def numeric(previous, current, *, atol=0.0, rtol=0.0):
    if previous is None or current is None:
        return dict(baseline=previous, current=current, comparable=False)
    same_shape = shape(previous) == shape(current)
    if not same_shape:
        return dict(comparable=False, baseline_shape=shape(previous), current_shape=shape(current))
    a, b = flattened(previous), flattened(current)
    differences = [y - x for x, y in zip(a, b)]
    output = dict(
        comparable=True,
        exactly_equal=previous == current,
        max_absolute_difference=max((abs(v) for v in differences), default=0.0),
        within_existing_gates=all(abs(y - x) <= atol + rtol * abs(x) for x, y in zip(a, b)),
        atol=atol,
        rtol=rtol,
    )
    if not isinstance(previous, list):
        output.update(baseline=previous, current=current, signed_difference=differences[0])
    return output


def role_for_start(path_index, total, index):
    if total == 4:
        return ("continuation", "pilot", "pooled", "alternative")[index]
    if path_index == 1 and total == 3:
        return ("pilot_and_equal_continuation", "pooled", "alternative")[index]
    return "deduplicated_start_index_" + str(index)


def start_summary(start, policy):
    gap = start.get("inner_gap")
    scale = start.get("inner_gap_scale")
    kkt = start.get("inner_kkt_residual")
    limit = None if scale is None else policy["inner_atol"] + policy["inner_rtol"] * scale
    return dict(
        status=start["status"],
        qualified=start["qualified"],
        objective=start.get("objective"),
        outer_iterations=start.get("outer_iterations"),
        admm_iterations=start.get("inner_iterations"),
        qp_calls=start.get("qp_calls"),
        backtracks=start.get("backtracks"),
        polish_iterations=start.get("qp_polish_iterations"),
        final_inner_gap=gap,
        allowed_gap=limit,
        gap_to_limit_ratio=None if limit is None or gap is None else gap / limit,
        final_kkt=kkt,
        arithmetic_gap_gate_passes=None if gap is None or limit is None else gap <= limit,
        arithmetic_kkt_gate_passes=None if kkt is None else kkt <= policy["inner_kkt_tol"],
        audit_calls=start.get("audit_calls"),
        qp_seconds=start.get("qp_seconds"),
    )


def compare(baseline, current):
    b, c = baseline["receipt"], current["receipt"]
    bp, cp = baseline["path"], current["path"]
    identity = dict(
        input_bytes=b["input_sha256"] == c["input_sha256"],
        input_model=baseline["input_identity"] == current["input_identity"],
        scientific_policy=b["policy"] == c["policy"],
        helpers=b["helpers"] == c["helpers"],
        pilot_and_graph=bp["pilot_and_graph"] == cp["pilot_and_graph"],
        literal_penalties=[r["lambda_value"] for r in bp["path_records"]]
        == [r["lambda_value"] for r in cp["path_records"]],
    )
    require(
        all(identity.values()),
        "Input/policy/helper/pilot/graph/path identity differs; cannot attribute a fixed-study regression",
    )
    candidates, common_failures, current_only, baseline_only = [], [], [], []
    for index, (old, new) in enumerate(zip(bp["path_records"], cp["path_records"])):
        detail = dict(
            index=index,
            lambda_value=old["lambda_value"],
            statuses={
                key: dict(
                    baseline=old.get(key), current=new.get(key), equal=old.get(key) == new.get(key)
                )
                for key in (
                    "raw_status",
                    "refit_status",
                    "search_complete",
                    "starts_attempted",
                    "starts_qualified",
                    "starts_unresolved",
                )
            },
            science={
                key: numeric(old.get(key), new.get(key), atol=1e-7, rtol=1e-10)
                for key in ("raw_objective", "score", "refit_gap")
            },
            starts=[],
        )
        be, ce = baseline["candidates"][index], current["candidates"][index]
        if "candidate_raw" in be and "candidate_raw" in ce:
            detail["raw_ccf"] = numeric(
                be["candidate_raw"]["raw_ccf"], ce["candidate_raw"]["raw_ccf"], atol=2e-5, rtol=1e-8
            )
            detail["memberships_equal_inferred_from_exact_raw_and_policy"] = (
                detail["raw_ccf"].get("exactly_equal") is True
            )
        if "candidate_refit" in be and "candidate_refit" in ce:
            detail["refit"] = {
                key: numeric(
                    be["candidate_refit"][key],
                    ce["candidate_refit"][key],
                    atol=2e-5 if key == "centers" else 1e-7,
                    rtol=1e-8 if key == "centers" else 1e-10,
                )
                for key in ("centers", "loss", "score", "gap")
            }
        old_starts, new_starts = old.get("starts", []), new.get("starts", [])
        detail["same_start_count"] = len(old_starts) == len(new_starts)
        for j in range(max(len(old_starts), len(new_starts))):
            os_ = old_starts[j] if j < len(old_starts) else None
            ns = new_starts[j] if j < len(new_starts) else None
            prior = None if os_ is None else start_summary(os_, b["policy"])
            now = None if ns is None else start_summary(ns, c["policy"])
            pair = dict(
                index=j,
                baseline_role=role_for_start(index, len(old_starts), j)
                if os_ is not None
                else None,
                current_role=role_for_start(index, len(new_starts), j) if ns is not None else None,
                baseline=prior,
                current=now,
            )
            if prior is not None and now is not None:
                pair["differences"] = {
                    key: numeric(prior[key], now[key])
                    for key in (
                        "objective",
                        "outer_iterations",
                        "admm_iterations",
                        "qp_calls",
                        "backtracks",
                        "polish_iterations",
                        "final_inner_gap",
                        "allowed_gap",
                        "final_kkt",
                    )
                }
                pair["qualified_equal"] = prior["qualified"] == now["qualified"]
                pair["status_equal"] = prior["status"] == now["status"]
            signature = dict(
                path_index=index,
                lambda_value=old["lambda_value"],
                start_index=j,
                baseline_role=pair["baseline_role"],
                current_role=pair["current_role"],
            )
            old_failed = prior is not None and not prior["qualified"]
            new_failed = now is not None and not now["qualified"]
            if old_failed and new_failed:
                common_failures.append(signature)
            elif new_failed:
                current_only.append(signature)
            elif old_failed:
                baseline_only.append(signature)
            detail["starts"].append(pair)
        candidates.append(detail)
    selected = dict(
        numerical={
            key: numeric(
                bp[key],
                cp[key],
                atol=2e-5 if key in ("raw_ccf", "refitted_ccf", "cluster_centers") else 1e-7,
                rtol=1e-8 if key in ("raw_ccf", "refitted_ccf", "cluster_centers") else 1e-10,
            )
            for key in (
                "selected_lambda",
                "raw_ccf",
                "refitted_ccf",
                "cluster_centers",
                "score",
                "raw_objective",
            )
        },
        cluster_labels_exactly_equal=bp["cluster_labels"] == cp["cluster_labels"],
        baseline_raw_qualified=bp["raw_qualified"],
        current_raw_qualified=cp["raw_qualified"],
    )
    roles = {}
    for name, data in (("baseline", baseline), ("current", current)):
        receipt, path = data["receipt"], data["path"]
        rows = path["path_records"]
        starts = [s for row in rows for s in row.get("starts", [])]
        roles[name] = dict(
            attempt=data["attempt"],
            job_id=data["job_id"],
            source_sha256=receipt["source"]["source_sha256"],
            status=receipt["status"],
            search_status=path["search_status"],
            final_export_and_publication_qualified=receipt["full_path"][
                "final_export_and_publication_qualified"
            ],
            planned_penalties=len(rows),
            qualified_raw_winners=sum(r["raw_status"] == "qualified" for r in rows),
            qualified_refits=sum(r["refit_status"] == "qualified" for r in rows),
            complete_penalties=sum(r.get("search_complete", False) for r in rows),
            starts_attempted=len(starts),
            starts_qualified=sum(s["qualified"] for s in starts),
            unresolved_start_statuses=dict(
                Counter(s["status"] for s in starts if not s["qualified"])
            ),
            fit_seconds=receipt["full_path"]["fit_seconds"],
            timings=path["timings"],
        )
    all_statuses_equal = all(
        all(v["equal"] for v in row["statuses"].values())
        and all(
            s.get("status_equal", False) and s.get("qualified_equal", False) for s in row["starts"]
        )
        for row in candidates
    )
    return dict(
        schema="clipp1d.cuda.mixed256.source-comparison.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        status="diagnostic_comparison_completed",
        scope="Read-only comparison of retained full adaptive-path evidence; no fitting, no tolerance changes, and no qualification inferred from selected winners.",
        identities_exact=identity,
        roles=roles,
        selected_candidate=selected,
        candidates=candidates,
        inherited_failure_signatures=common_failures,
        current_only_failure_signatures=current_only,
        baseline_only_failure_signatures=baseline_only,
        all_per_candidate_and_start_statuses_equal=all_statuses_equal,
        both_full_paths_incomplete=all(v["search_status"] == "incomplete" for v in roles.values()),
        interpretation="Failures at matched deterministic start positions demonstrate inherited coverage failure when both sources fail; any source-specific failure remains separately listed. Comparable winners do not certify the full path.",
        limitations=[
            "This full-path comparison is distinct from the64-node explicit identical shared-pilot/fixed-lambda replays.",
            "Per-start pairing uses deterministic source order after exact duplicate removal; missing or differing start counts are explicit.",
            "Per-start raw vectors were not recorded. Candidate raw vectors and refit centers are compared from the bound journal.",
            "QP gaps and iteration counts may change while both sources satisfy the same certificate gates; failed-start objective values are diagnostic, not admitted fit states.",
            "The reported inner_kkt_qualified boolean in these sources is the joint QP result. This report recomputes arithmetic gap/KKT comparisons separately.",
            "Full-fit timings are single instrumented observations, not repeated throughput evidence. Neither incomplete path admits512 qualification.",
        ],
        source_artifacts=baseline["source_artifacts"] | current["source_artifacts"],
        comparison_script_sha256=sha(Path(__file__)),
    )


def main():
    output = BASE / "baseline256-g/DIAGNOSIS.json"
    require(
        not output.exists() and not output.is_symlink(),
        "Existing diagnosis must not be overwritten",
    )
    report = compare(read_attempt("baseline"), read_attempt("current"))
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    require(sha(output) == hashlib.sha256(payload).hexdigest(), "Diagnosis readback differs")
    print(
        json.dumps(
            dict(
                path=str(output),
                sha256=sha(output),
                bytes=len(payload),
                inherited_failures=len(report["inherited_failure_signatures"]),
                current_only_failures=len(report["current_only_failure_signatures"]),
                baseline_only_failures=len(report["baseline_only_failure_signatures"]),
            )
        )
    )


if __name__ == "__main__":
    main()
