"""Small, self-contained contracts for the b10e5c6 compaction.

Run with an installed, source-bound build: python -m unittest discover -s tests.
No baseline checkout, pytest, or generated fixture is needed. Differential
whole-bank and high-copy-number checks live in compare_sources.py.
"""

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from clipp import _io, api, selection, versions
from clipp._candidate_store import CandidateStore, CandidateStoreError
from clipp._proposals import _integers, iter_chain_proposals
from clipp.config import FitConfig
from clipp.output import Result, load_result
from clipp.refinement import fixed_center_chain_partition, polish_chain_partition
from clipp.verify import verify_run


# Deliberately literal and independent of the shared production declarations.
FINALIST_FIELDS = {
    "requested_k",
    "replicate",
    "candidate_id",
    "candidate_kind",
    "parent_requested_k",
    "parent_replicate",
    "parent_partition_sha256",
    "proposal_partition_sha256",
    "partition_sha256",
    "num_clusters",
    "bic",
    "log_likelihood",
    "conditional_log_likelihood",
    "weight_optimality_gap",
    "weight_active_score_gap",
    "proposal_cuts",
    "partition_parameters",
}
INTEGER_FIELDS = {
    "requested_k",
    "replicate",
    "candidate_id",
    "parent_requested_k",
    "parent_replicate",
    "num_clusters",
}
SCORE_FIELDS = {
    "bic",
    "log_likelihood",
    "conditional_log_likelihood",
    "weight_optimality_gap",
    "weight_active_score_gap",
}


def labels_by_intervals(order, cuts, block_labels=None):
    """Independent interval-loop oracle, not the production searchsorted code."""
    result = np.empty(len(order), dtype=np.int64)
    boundaries = [0, *cuts, len(order)]
    for block, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        result[np.asarray(order)[start:stop]] = block if block_labels is None else block_labels[block]
    return result


def digest(order, cuts):
    return hashlib.sha256(
        np.asarray(order, dtype="<i8").tobytes() + np.asarray(cuts, dtype="<i8").tobytes()
    ).hexdigest()


def candidate(**changes):
    record = dict(
        requested_k=2,
        status="scored",
        candidate_kind="native",
        parent_requested_k=2,
        parent_replicate=1,
        parent_partition_sha256="raw",
        proposal_partition_sha256="proposal",
        partition_sha256="final",
        chain_cuts="4",
        num_clusters=2,
        bic=1.25,
        publication_eligible=True,
        partition_parameters='{"cuts":[4],"block_labels":[0,1],"centers":[0.7,0.2],"weights":[0.6,0.4]}',
    )
    return {**record, **changes}


class RepresentationTests(unittest.TestCase):
    def test_partition_round_trips_and_nonmonotone_centers(self):
        rng = np.random.default_rng(84712)
        for n in (1, 2, 7, 19):
            for _ in range(12):
                order = rng.permutation(n)
                for cuts in (
                    [],
                    list(range(1, n)),
                    sorted(rng.choice(np.arange(1, n), size=(n - 1) // 2, replace=False).tolist()),
                ):
                    q = len(cuts) + 1
                    visits = rng.permutation(q)
                    labels = labels_by_intervals(order, cuts, visits)
                    fitted = dict(
                        labels=labels,
                        centers=rng.permutation(q).astype(float) / q,
                        cluster_weights=np.full(q, 1 / q),
                    )
                    parameters = json.loads(selection._parameters(fitted, order))
                    self.assertEqual(parameters["cuts"], cuts)
                    self.assertEqual(parameters["block_labels"], visits.tolist())
                    cache = selection._ChainRefitCache(range(n), order)
                    key = cache.partition_key(labels)
                    cache.put_partition(key, fitted)
                    restored = cache.get_partition(key)
                    for name in fitted:
                        np.testing.assert_array_equal(restored[name], fitted[name])
                    restored["labels"][:] = -1
                    restored["centers"][:] = -1
                    np.testing.assert_array_equal(cache.get_partition(key)["labels"], labels)
                    result = Result.__new__(Result)
                    result._order = order
                    result._parameters = {(1, 0): parameters}
                    result.fits = pd.DataFrame(
                        [dict(requested_k=q, replicate=1, candidate_id=0, selected_for_k=True)]
                    )
                    np.testing.assert_array_equal(result.partition(q)["labels"], labels)
                    # Baseline lacks this new helper; the same call sites above
                    # remain exercised before and after it is introduced.
                    if hasattr(_io, "partition_labels"):
                        np.testing.assert_array_equal(_io.partition_labels(order, cuts, visits), labels)
                        np.testing.assert_array_equal(
                            _io.partition_labels(order, cuts), labels_by_intervals(order, cuts)
                        )

    def test_integer_conversion_and_exception_contract(self):
        accepted = [
            (3, None, [3]),
            ([True, False], 2, [1, 0]),
            (["2", "-1"], 2, [2, -1]),
            (np.array([0.0, 4.0]), 2, [0, 4]),
            ([2**53 + 1], 1, [2**53]),
        ]
        invalid = [
            ([], None),
            ([[1, 2]], None),
            ([1.5], None),
            ([np.nan], None),
            ([np.inf], None),
            ([1, 2], 1),
            (["bad"], None),
        ]
        for validator in (_integers, selection._integer_vector):
            for value, length, expected in accepted:
                actual = validator(value, "fixture", length)
                self.assertEqual(actual.dtype, np.dtype("int64"))
                np.testing.assert_array_equal(actual, expected)
            for value, length in invalid:
                with self.assertRaises(ValueError):
                    validator(value, "fixture", length)
            with self.assertRaises(TypeError):
                validator([object()], "fixture")

    def test_exact_parameter_text_and_literal_finalist_schema(self):
        record = dict(
            labels=np.array([1, 1, 0]), centers=np.array([0.25, 0.75]), cluster_weights=np.array([0.25, 0.75])
        )
        self.assertEqual(
            selection._parameters(record, np.arange(3)),
            '{"cuts":[2],"block_labels":[1,0],"centers":[0.25,0.75],"weights":[0.25,0.75]}',
        )
        if hasattr(versions, "FIT_FIELDS"):
            self.assertEqual(
                set(versions.FIT_FIELDS), FINALIST_FIELDS - {"proposal_cuts", "partition_parameters"}
            )
            self.assertEqual(set(versions.FIT_INTEGER_FIELDS), INTEGER_FIELDS)
            self.assertEqual(set(versions.FIT_SCORE_FIELDS), SCORE_FIELDS)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_exact_interning_missing_fields_scalars_and_detached_updates(self):
        text = ' {"centers": [-0.0,5e-324,0.9999999999999999]} '
        with CandidateStore(directory=self.directory, transaction_rows=2) as store:
            first = store.append(
                candidate(partition_parameters=text, bic=np.float64(-0.0), parent_requested_k=np.int64(2))
            )
            store.append(candidate(partition_parameters=text))
            store.append(candidate(partition_parameters=text, chain_cuts="5"))
            store.append({"requested_k": 2, "status": "refit_failed", "error": "fixture"})
            read = store.get(1, first["candidate_id"])
            self.assertEqual(read["partition_parameters"], text)
            self.assertEqual(struct.pack("d", read["bic"]), struct.pack("d", -0.0))
            for table, count in (("parameters", 1), ("sources", 1), ("proposals", 2)):
                self.assertEqual(store._db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0], count)
            for name in (
                "partition_parameters",
                "parent_requested_k",
                "chain_cuts",
                "bic",
                "publication_eligible",
            ):
                self.assertNotIn(name, store.get(1, 3))
            read["selected"] = True
            first["selected"] = True
            self.assertFalse(store.get(1, 0)["selected"])
            store.update(1, 0, {"selected": True})
            self.assertIs(store.get(1, 0)["selected"], True)
            with self.assertRaises(TypeError):
                store.append(candidate(selected=1))
            with self.assertRaises(TypeError):
                store.append(candidate(partition_parameters={"cuts": [1]}))
            with self.assertRaises(ValueError):
                store.update(1, 0, {"candidate_id": 10})
            self.assertGreater(store.peak_scratch_bytes, 0)
            path = store.path
        self.assertFalse(path.exists())

    def test_ids_bounded_prefix_ranking_ancestry_and_visited(self):
        with CandidateStore(directory=self.directory, transaction_rows=1) as store:
            store.append(candidate(status="refit_failed"))
            store.append(candidate(bic=2.0))
            store.append(candidate(bic=2.0))
            store.append(candidate(bic=0.0, publication_eligible=False))
            store.append(candidate(), replicate=2)
            self.assertEqual(store.next_id(1), 4)
            self.assertEqual(store.next_id(2), 1)
            with self.assertRaises(ValueError):
                store.append(candidate(candidate_id=1))
            cursor = store.iter_rows(replicate=1)
            self.assertEqual(next(cursor)["candidate_id"], 0)
            store.append(candidate(bic=3.0))
            self.assertEqual([row["candidate_id"] for row in cursor], [1, 2, 3])
            self.assertEqual(store.best(1, 2)["candidate_id"], 1)
            self.assertEqual(store.best(1, 2, eligible=False)["candidate_id"], 3)
            self.assertEqual(store.find_refinement_parent(1, 2, 4, 2, "raw", "final")["candidate_id"], 1)
            self.assertIsNone(store.find_refinement_parent(1, 2, 1, 2, "raw", "final"))
            self.assertEqual(store.count(replicate=1), 5)
            self.assertEqual(len(store), 6)
            self.assertTrue(store.visited_add("one", (2, (3, 10))))
            self.assertFalse(store.visited_add("one", (2, (3, 10))))
            self.assertTrue(store.visited_add("two", (2, (3, 10))))
            store.visited_discard("one", (2, (3, 10)))
            self.assertFalse(store.visited_contains("one", (2, (3, 10))))
            self.assertTrue(store.visited_contains("two", (2, (3, 10))))

    def test_sqlite_failure_is_terminal_and_retains_readable_evidence(self):
        store = CandidateStore(directory=self.directory, transaction_rows=1)
        store.append(candidate())
        pages = store._db.execute("PRAGMA page_count").fetchone()[0]
        store._db.execute(f"PRAGMA max_page_count={pages}")
        with self.assertRaisesRegex(CandidateStoreError, "full"):
            store.append(candidate(partition_parameters="x" * 1_000_000))
        with self.assertRaisesRegex(CandidateStoreError, "previously failed"):
            store.append(candidate())
        store.close(remove=False)
        with sqlite3.connect(store.path) as connection:
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")


class SearchTests(unittest.TestCase):
    def test_duplicate_routes_canonical_order_and_exact_fingerprints(self):
        order = np.array([3, 0, 5, 1, 4, 2])
        seeds = [dict(requested_k=k, labels=labels_by_intervals(order, [3], [9, 2])) for k in (5, 2, 5)]
        proposals = list(iter_chain_proposals(seeds, order, [1, 2, 5]))
        self.assertEqual([cuts for cuts, _ in proposals], [(), (3,)])
        self.assertEqual(list(proposals[1][1]), [5, 2])
        for cuts, routes in proposals:
            for k, route in routes.items():
                self.assertEqual(
                    route,
                    dict(
                        requested_k=k,
                        parent_requested_k=k if cuts else 2,
                        parent_replicate=1,
                        candidate_kind="native" if cuts else "adjacent_coarsening",
                        parent_partition_sha256=digest(order, [3]),
                    ),
                )

    def test_unsupported_merge_and_exact_tie_reconciliation(self):
        order = np.array([3, 7, 0, 6, 1, 5, 2, 4])

        def fitted(cuts, weights, bic, conditional=-38.25):
            q = len(weights)
            return dict(
                labels=labels_by_intervals(order, cuts, list(reversed(range(q)))),
                centers=np.linspace(0.8, 0.2, q),
                cluster_weights=np.array(weights),
                num_clusters=q,
                bic=float(bic),
                log_likelihood=-50.0,
                conditional_log_likelihood=conditional,
                weight_optimality_gap=1e-8,
                weight_active_score_gap=1e-10,
            )

        def append(store, value, cuts, kind, parent_k):
            final = tuple(np.flatnonzero(np.diff(value["labels"][order])) + 1)
            return store.append(
                candidate(
                    requested_k=4,
                    candidate_kind=kind,
                    parent_requested_k=parent_k,
                    parent_partition_sha256=digest(order, cuts),
                    proposal_partition_sha256=digest(order, cuts),
                    partition_sha256=digest(order, final),
                    chain_cuts=",".join(map(str, cuts)),
                    num_input_blocks=len(cuts) + 1,
                    partition_parameters=selection._parameters(value, order),
                    publication_eligible=bool(np.all(value["cluster_weights"] > 0)),
                    active_mixture_components=int(np.count_nonzero(value["cluster_weights"])),
                    candidate_search_version=versions.IDENTITIES["candidate_search_version"],
                    **{name: value[name] for name in (*SCORE_FIELDS, "num_clusters")},
                )
            )

        for bic in (np.nextafter(100.0, -np.inf), 100.0, np.nextafter(100.0, np.inf)):
            with self.subTest(bic=bic), CandidateStore() as store:
                earlier = fitted((3, 5), [0.4, 0.2, 0.4], bic)
                repair = fitted((3, 5), [0.4, 0.2, 0.4], 100.0, np.nextafter(-38.25, np.inf))
                unsupported = fitted((1, 3, 5), [0.4, 0.2, 0.4, 0.0], 90.0)
                append(store, earlier, (3, 5, 6), "adjacent_coarsening", 5)
                native = append(store, unsupported, (1, 3, 5), "native", 4)
                before = list(store.iter_rows())
                expected_id = 0 if bic <= 100 else 2
                retained = earlier if expected_id == 0 else repair
                expected_calls = [((3, 5), repair), ((3, 5, 6) if expected_id == 0 else (3, 5), retained)]
                calls, polishes = [], []

                def refit(model, labels, cache, chain_order):
                    cuts = tuple(np.flatnonzero(np.diff(labels[chain_order])) + 1)
                    expected_cuts, value = expected_calls[len(calls)]
                    self.assertEqual(cuts, expected_cuts)
                    calls.append(cuts)
                    return copy.deepcopy(value)

                def polish(model, value, chain_order, callback):
                    polishes.append(tuple(np.flatnonzero(np.diff(value["labels"][chain_order])) + 1))
                    return dict(result=copy.deepcopy(value), diagnostics={"status": "fixture_stable"})

                search = dict(
                    replicate=1, winners={4: {**native, "result": unsupported}}, candidates=store, stats={}
                )
                with (
                    patch.object(selection, "refit_partition", refit),
                    patch.object(selection, "polish_chain_partition", polish),
                ):
                    actual = selection.refine_chain_search(
                        range(8), search, [dict(requested_k=4, labels=unsupported["labels"])], order, {}
                    )
                self.assertEqual(calls, [item[0] for item in expected_calls])
                self.assertEqual(polishes, [(1, 3, 5), (3, 5)])
                self.assertEqual(len(store), 3)
                self.assertEqual(store.get(1, 2)["candidate_kind"], "unsupported_component_coarsening")
                self.assertEqual(actual["winners"][4]["candidate_id"], expected_id)
                for name in ("labels", "centers", "cluster_weights"):
                    self.assertEqual(actual["winners"][4]["result"][name].tobytes(), retained[name].tobytes())
                for row in before:
                    after = store.get(1, row["candidate_id"])
                    for name in (
                        *SCORE_FIELDS,
                        "partition_parameters",
                        "partition_sha256",
                        "proposal_partition_sha256",
                        "chain_cuts",
                    ):
                        self.assertEqual(after[name], row[name])
                for row in store.iter_rows():
                    self.assertEqual(row["selected_for_k"], row["candidate_id"] == expected_id)
                    self.assertEqual(row["selected_for_replicate_k"], row["candidate_id"] == expected_id)


class QuadraticModel:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def __len__(self):
        return len(self.values)

    def log_likelihood(self, center):
        return -((self.values - center) ** 2)

    def refit(self, labels):
        return dict(
            labels=labels.copy(),
            centers=np.array([self.values[labels == k].mean() for k in np.unique(labels)]),
        )


class Progress:
    """Capture logical ordering and counters without wall-clock measurements."""

    def __init__(self):
        self.events = []

    def update(self, name, **fields):
        self.events.append((name, fields))

    @contextmanager
    def timer(self, name):
        self.events.append(("timer_started", name))
        try:
            yield
        finally:
            self.events.append(("timer_completed", name))

    @contextmanager
    def context(self, **fields):
        self.events.append(("context_started", fields))
        try:
            yield
        finally:
            self.events.append(("context_completed", fields))


class RefinementTests(unittest.TestCase):
    def test_all_completion_branches_history_events_and_calls(self):
        for branch in ("stable", "failure", "decrease", "budget", "tolerance", "accepted_then_stable"):
            with self.subTest(branch=branch):
                model = QuadraticModel([1, 1, 1, 0, 0, 0])
                initial = model.refit(
                    np.array([0, 0, 0, 1, 1, 1]) if branch == "stable" else np.array([0, 0, 1, 1, 1, 1])
                )
                calls, progress = [], Progress()

                def refit(labels):
                    calls.append(labels.tolist())
                    if branch == "failure":
                        raise RuntimeError("fixture refit failure")
                    if branch == "decrease":
                        return dict(labels=labels, centers=np.array([2.0, 3.0]))
                    return model.refit(labels)

                result = polish_chain_partition(
                    model,
                    initial,
                    np.arange(6),
                    refit,
                    max_iterations=1 if branch == "budget" else 3,
                    mean_loglik_tolerance=0.125 if branch == "tolerance" else 0.0,
                    progress=progress,
                )
                status = {
                    "stable": "fixed_center_boundaries_stable",
                    "failure": "refit_failed",
                    "decrease": "refit_conditional_decrease",
                    "budget": "iteration_budget",
                    "tolerance": "conditional_improvement_tolerance",
                    "accepted_then_stable": "fixed_center_boundaries_stable",
                }[branch]
                entry = dict(
                    iteration=1,
                    num_clusters=2,
                    previous_conditional_log_likelihood=0.0 if branch == "stable" else -0.75,
                    fixed_center_conditional_log_likelihood=0.0 if branch == "stable" else -0.1875,
                    boundaries_changed=branch != "stable",
                )
                if branch == "failure":
                    entry.update(status="refit_failed", error="fixture refit failure")
                elif branch == "stable":
                    entry["status"] = status
                else:
                    entry.update(
                        refit_conditional_log_likelihood=-30.0 if branch == "decrease" else 0.0,
                        refit_num_clusters=2,
                        status="refit_conditional_decrease" if branch == "decrease" else "accepted",
                    )
                    if branch != "decrease":
                        entry["mean_loglik_improvement"] = 0.125
                expected = [entry]
                if branch == "accepted_then_stable":
                    expected.append(
                        dict(
                            iteration=2,
                            num_clusters=2,
                            previous_conditional_log_likelihood=0.0,
                            fixed_center_conditional_log_likelihood=0.0,
                            boundaries_changed=False,
                            status="fixed_center_boundaries_stable",
                        )
                    )
                self.assertEqual(result["history"], expected)
                self.assertEqual(
                    result["diagnostics"],
                    dict(
                        method="ordered_conditional_chain_boundary_refinement_v1",
                        status=status,
                        iterations=len(expected),
                        max_iterations=1 if branch == "budget" else 3,
                        mean_loglik_tolerance=0.125 if branch == "tolerance" else 0.0,
                        initial_num_clusters=2,
                        final_num_clusters=2,
                        initial_conditional_log_likelihood=0.0 if branch == "stable" else -0.75,
                        conditional_log_likelihood=-0.75 if branch in ("failure", "decrease") else 0.0,
                        conditional_log_likelihood_improvement=0.75
                        if branch in ("budget", "tolerance", "accepted_then_stable")
                        else 0.0,
                        fixed_chain=True,
                        unconstrained_reassignment=False,
                        constrained_optimum_certified=False,
                    ),
                )
                events = []
                for item in expected:
                    iteration = item["iteration"]
                    events.extend(
                        [
                            ("polish_iteration_started", {"refinement_iteration": iteration}),
                            ("timer_started", "boundary_dynamic_programming"),
                            ("timer_completed", "boundary_dynamic_programming"),
                        ]
                    )
                    if item["boundaries_changed"]:
                        events.extend(
                            [
                                ("context_started", {"refinement_iteration": iteration}),
                                ("context_completed", {"refinement_iteration": iteration}),
                            ]
                        )
                    events.append(
                        (
                            "polish_iteration_completed",
                            {"refinement_iteration": iteration, "status": item["status"]},
                        )
                    )
                self.assertEqual(progress.events, events)
                self.assertEqual(calls, [] if branch == "stable" else [[0, 0, 0, 1, 1, 1]])
                if branch in ("failure", "decrease", "stable"):
                    self.assertIs(result["result"], initial)
                else:
                    np.testing.assert_array_equal(result["result"]["labels"], calls[0])

    def test_accepted_equal_score_retains_best_unless_component_count_drops(self):
        model = QuadraticModel([0, 0, 0, 0])
        initial = dict(labels=np.array([0, 0, 0, 1]), centers=np.zeros(2))
        for merge in (False, True):

            def refit(labels):
                return dict(
                    labels=np.zeros(4, dtype=int) if merge else labels, centers=np.zeros(1 if merge else 2)
                )

            result = polish_chain_partition(model, initial, np.arange(4), refit)
            self.assertEqual(result["diagnostics"]["status"], "conditional_improvement_tolerance")
            self.assertEqual(result["history"][0]["status"], "accepted")
            self.assertEqual(result["diagnostics"]["final_num_clusters"], 1 if merge else 2)
            if not merge:
                self.assertIs(result["result"], initial)

    def test_invalid_boundary_move_and_decreasing_dp_are_fatal(self):
        model = QuadraticModel([1, 1, 1, 0, 0, 0])
        initial = model.refit(np.array([0, 0, 1, 1, 1, 1]))
        with self.assertRaisesRegex(ValueError, "only preserve boundaries"):
            polish_chain_partition(
                model, initial, np.arange(6), lambda _: model.refit(np.array([0, 1, 1, 1, 1, 1]))
            )
        with (
            patch(
                "clipp.refinement.fixed_center_chain_partition",
                return_value={"conditional_log_likelihood": -2.0},
            ),
            self.assertRaisesRegex(RuntimeError, "decreased conditional"),
        ):
            polish_chain_partition(model, initial, np.arange(6), model.refit)

    def test_fixed_center_ties_and_nonmonotone_centers(self):
        np.testing.assert_array_equal(fixed_center_chain_partition(np.zeros((7, 3)))["cuts"], [1, 2])
        model = QuadraticModel([0.75, 0.75, 0.75, 0.25, 0.25, 0.5, 0.5])
        initial = model.refit(np.array([0, 0, 1, 1, 2, 2, 2]))
        result = polish_chain_partition(model, initial, np.arange(7), model.refit)["result"]
        np.testing.assert_array_equal(result["centers"][result["labels"]], model.values)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.inputs = [self.root / name for name in ("snv.tsv", "cna.tsv", "purity.txt")]
        self.inputs[0].write_text(
            "chromosome_index\tposition\talt_count\tref_count\n1\t10\t2\t18\n1\t20\t3\t17\n1\t30\t7\t13\n1\t40\t8\t12\n"
        )
        self.inputs[1].write_text(
            "chromosome_index\tstart_position\tend_position\tmajor_cn\tminor_cn\ttotal_cn\n1\t1\t100\t1\t1\t2\n"
        )
        self.inputs[2].write_text("1\n")
        for target, value in (
            ("clipp.api.tempfile.gettempdir", lambda: str(self.scratch)),
            ("clipp.api._progress", lambda _: None),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def fit(self, **config):
        output = self.root / "result"
        api.fit(*self.inputs, output, config=FitConfig(device="cpu", max_clusters=2, **config))
        return output

    def assert_failed(self):
        self.assertFalse((self.root / "result").exists())
        self.assertFalse((self.root / ".result.lock").exists())
        attempts = list(self.root.glob(".result.inprogress.*"))
        self.assertEqual(len(attempts), 1)
        self.assertFalse(list(attempts[0].rglob("COMPLETE.json")))
        failure = json.loads((attempts[0] / "FAILURE.json").read_text())
        self.assertEqual(failure["status"], "failed")
        if failure["candidate_evidence"] is not None:
            self.assertTrue(Path(failure["candidate_evidence"]).is_file())
        return failure

    def test_literal_saved_schema_replicates_and_verified_publication(self):
        output = self.fit(subsample_size=3, replicates=2, window_size=1.0, seed=7)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["backend_actual"], "cpu")
        self.assertEqual(manifest["output_schema_version"], 4)
        self.assertEqual(len(manifest["fits"]), 4)
        for row in manifest["fits"]:
            self.assertEqual(set(row), FINALIST_FIELDS)
            self.assertEqual(set(row["partition_parameters"]), {"cuts", "block_labels", "centers", "weights"})
            for name in INTEGER_FIELDS:
                self.assertIs(type(row[name]), int)
        self.assertEqual(
            {p.name for p in (output / "final_result").iterdir()}, {"mutations.tsv", "clusters.tsv"}
        )
        result = load_result(output)
        self.assertNotIn("mixture_weight", result.clusters)
        self.assertNotIn("multiplicity_probability", result.mutations)
        for capacity in (1, 2):
            for replicate in (1, 2):
                self.assertEqual(len(result.partition(capacity, replicate)["labels"]), 4)
        self.assertFalse(list(self.scratch.iterdir()))
        self.assertEqual(verify_run(output)["selected_k"], manifest["selected_k"])
        # Reseal the changed manifest: independent arithmetic must reject it,
        # even when the ordinary file-integrity check has been recomputed.
        manifest["fits"][0]["bic"] += 100.0
        path = output / "manifest.json"
        path.write_text(json.dumps(manifest))
        complete_path = output / "COMPLETE.json"
        complete = json.loads(complete_path.read_text())
        complete["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        complete_path.write_text(json.dumps(complete))
        with self.assertRaises(ValueError):
            verify_run(output)

    def test_sqlite_disk_full_prevents_publication(self):
        class FullStore(CandidateStore):
            def append(self, record, **kwargs):
                pages = self._db.execute("PRAGMA page_count").fetchone()[0]
                self._db.execute("PRAGMA max_page_count=" + str(pages))
                return super().append({**record, "exhaust_page_budget": "x" * 1_000_000}, **kwargs)

        with (
            patch.object(api, "CandidateStore", FullStore),
            self.assertRaisesRegex(CandidateStoreError, "full"),
        ):
            self.fit()
        self.assertEqual(self.assert_failed()["type"], "CandidateStoreError")

    def test_cleanup_failure_prevents_completion_after_verification(self):
        class CloseErrorStore(CandidateStore):
            def close(self, *, remove=None):
                if remove is None:
                    super().close(remove=False)
                    raise OSError("fixture scratch cleanup failure")
                super().close(remove=remove)

        with (
            patch.object(api, "CandidateStore", CloseErrorStore),
            self.assertRaisesRegex(OSError, "scratch cleanup failure"),
        ):
            self.fit()
        failure = self.assert_failed()
        self.assertEqual(failure["type"], "OSError")
        self.assertIn("independent_verification", failure["stage_seconds"])

    def test_verification_error_is_not_masked_by_secondary_cleanup_error(self):
        class CloseErrorStore(CandidateStore):
            def close(self, *, remove=None):
                super().close(remove=remove)
                if remove is False:
                    raise CandidateStoreError("fixture secondary cleanup failure")

        with (
            patch.object(api, "CandidateStore", CloseErrorStore),
            patch("clipp.verify.verify_run", side_effect=ValueError("fixture primary verification failure")),
            self.assertRaisesRegex(ValueError, "primary verification failure") as caught,
        ):
            self.fit()
        self.assertTrue(any("secondary cleanup failure" in note for note in caught.exception.__notes__))
        self.assertEqual(self.assert_failed()["error"], "fixture primary verification failure")


if __name__ == "__main__":
    unittest.main()
