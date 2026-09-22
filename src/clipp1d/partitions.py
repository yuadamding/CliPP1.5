"""Bounded chain-contiguous partition proposals and qualified likelihood refits.

Direct candidates never acquire the raw fusion state's certificate or lambda.
All acceptance uses the original constrained scalar refit and partition score.
The fixed proposal budgets bound overhead, not global partition optimality.
"""
from collections import OrderedDict
from time import perf_counter

import numpy as np
from scipy.special import gammaln

from .model import evaluate, loss
from .policy import Policy
from .scalar import minimize_block
from .selection import refit_partition
from .types import ClonalConstraintInfeasibleError, NumericalQualificationError
from .ward import adjacent_ward_cuts

PROPOSAL_POLICY = dict(max_blocks=12, refinement_seeds=1, boundary_sweeps=4,
                        partition_budget=256, interval_cache_entries=512,
                        score_roundoff_margin=1e-8, weights='max(current positive curvature majorizer,1)',
                        seed_sources=['qualified_marginal_pilot','selected_qualified_raw'],
                        boundary_proposals='best fixed-center feasible cut and current cut minus/plus one',
                        acceptance='score decrease greater than twice both refit gaps plus roundoff margin')


class BudgetReached(Exception):
    pass


def improves(candidate, current):
    uncertainty = 2 * (candidate.gap + current.gap) + PROPOSAL_POLICY['score_roundoff_margin']
    return candidate.score < current.score - uncertainty


class IntervalRefitter:
    """Bounded scalar cache scoped to one immutable ordered model and policy."""
    def __init__(self, model, policy=Policy()):
        self.model, self.policy = model, policy
        self.cache = OrderedDict()
        self.partitions = {}
        self.origins = {}
        self.seed_origins = {}
        self.at_one = loss(model, np.minimum(model.upper, 1))
        self.counters = dict(partition_attempts=0, interval_fits=0, interval_cache_hits=0,
                             scalar_evaluations=0, scalar_rows=0, peak_cache_entries=0,
                             boundary_scans=0, boundary_likelihood_rows=0)
        self.failures = []

    def interval(self, start, stop):
        key = (start, stop)
        if key in self.cache:
            self.counters['interval_cache_hits'] += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.counters['interval_fits'] += 1
        self.counters['scalar_rows'] += stop-start
        scalar = minimize_block(self.model.subset(np.arange(start, stop)), self.policy)
        self.counters['scalar_evaluations'] += scalar.evaluations + scalar.bound_evaluations
        if not scalar.qualified:
            raise NumericalQualificationError('Proposal interval did not qualify', interval=key, gap=scalar.optimality_gap)
        # Cache scalars only; no copied per-interval model or row vectors persist.
        value = scalar
        self.cache[key] = value
        if len(self.cache) > PROPOSAL_POLICY['interval_cache_entries']:
            self.cache.popitem(last=False)
        self.counters['peak_cache_entries'] = max(self.counters['peak_cache_entries'], len(self.cache))
        return value

    def fit(self, cuts):
        cuts = tuple(cuts)
        if (not cuts or cuts[0] != 0 or cuts[-1] != len(self.model) or
                any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in cuts) or
                any(a >= b for a, b in zip(cuts[:-1], cuts[1:]))):
            raise ValueError('Cuts must partition the chain into occupied integer intervals')
        if cuts in self.partitions:
            return self.partitions[cuts]
        if self.counters['partition_attempts'] >= PROPOSAL_POLICY['partition_budget']:
            raise BudgetReached()
        self.counters['partition_attempts'] += 1
        try:
            result = refit_partition(self.model, cuts, self.policy, interval_solver=self.interval)
        except (NumericalQualificationError, ClonalConstraintInfeasibleError) as error:
            self.failures.append(dict(cuts=cuts,error=type(error).__name__,message=str(error)))
            result = None
        self.partitions[cuts] = result
        return result


def boundary_candidates(refitter, current, boundary):
    """Rank all feasible cuts at fixed centers in linear row work.

    Prefix arithmetic is proposal ranking only. Every proposed move receives
    a fresh qualified likelihood refit and exact score acceptance afterwards.
    No monotonic CCF constraint or nonadjacent reassignment is introduced.
    """
    start, middle, stop = current.cuts[boundary-1:boundary+2]
    if stop-start < 3:
        return []
    left, right = current.centers[boundary-1:boundary+1]
    model = refitter.model.subset(np.arange(start, stop))
    n = len(model)
    left_rows = loss(model, np.full(n, left))
    right_rows = loss(model, np.full(n, right))
    left_invalid = (left < model.lower) | (left > model.upper)
    right_invalid = (right < model.lower) | (right > model.upper)
    left_count = np.cumsum(left_invalid)[:-1]
    right_count = np.cumsum(right_invalid[::-1])[::-1][1:]
    feasible = (left_count == 0) & (right_count == 0)
    left_loss = np.cumsum(left_rows.astype(np.longdouble))[:-1]
    right_loss = np.cumsum(right_rows[::-1].astype(np.longdouble))[::-1][1:]
    sizes = np.arange(1,n)
    allocation = -1.4 * (gammaln(sizes+1) + gammaln(n-sizes+1))
    scores = 2*(left_loss+right_loss) + allocation
    scores[~feasible] = np.inf
    proposed = [start+int(np.argmin(scores))+1] if np.any(feasible) else []
    proposed += [middle-1,middle+1]
    refitter.counters['boundary_scans'] += 1
    refitter.counters['boundary_likelihood_rows'] += 2*n
    return sorted({p for p in proposed if start < p < stop and p != middle})


def refine(refitter, seed, lineage):
    current = seed
    history = []
    stopped = 'sweep_budget'
    for sweep in range(PROPOSAL_POLICY['boundary_sweeps']):
        changed = False
        for boundary in range(1,len(current.cuts)-1):
            winner = current
            for cut in boundary_candidates(refitter,current,boundary):
                cuts = list(current.cuts)
                cuts[boundary] = cut
                refitter.origins.setdefault(tuple(cuts), 'boundary_refinement')
                refitter.seed_origins.setdefault(tuple(cuts), lineage)
                candidate = refitter.fit(cuts)
                if candidate is not None and improves(candidate,winner):
                    winner = candidate
            if winner is not current:
                history.append(dict(sweep=sweep,boundary=boundary,before=current.cuts,after=winner.cuts,
                                    score_before=current.score,score_after=winner.score))
                current = winner
                changed = True
        if not changed:
            stopped = 'no_improving_planned_move'
            break
    return current, dict(seed=lineage,seed_cuts=seed.cuts,moves=history,stop=stopped)


def propose_partitions(model, pilot_phi, raw_phi, baseline, policy=Policy()):
    started = perf_counter()
    fitter = IntervalRefitter(model, policy)
    fitter.partitions[baseline.cuts] = baseline
    fitter.counters['partition_attempts'] = 1
    best = baseline
    origins = fitter.origins
    origins[baseline.cuts] = 'production_path'
    work, refinements = [], []
    exhausted = False
    try:
        for source, phi in [('pilot',pilot_phi),('selected_raw',raw_phi)]:
            weights = np.maximum(evaluate(model,phi,derivatives=True).curvature,1.)
            cuts_list, counters = adjacent_ward_cuts(phi,weights,PROPOSAL_POLICY['max_blocks'])
            work.append(dict(source=source,**counters))
            for cuts in cuts_list:
                cuts = tuple(cuts)
                origins.setdefault(cuts,'adjacent_ward_'+source)
                candidate = fitter.fit(cuts)
                if candidate is not None and improves(candidate,best):
                    best = candidate
        before_refinement = best
        seeds = sorted((v for v in fitter.partitions.values() if v is not None),
                       key=lambda v:(v.score,len(v.cuts),v.cuts))[:PROPOSAL_POLICY['refinement_seeds']]
        for seed in seeds:
            refined, history = refine(fitter,seed,origins[seed.cuts])
            refinements.append(history)
            origins.setdefault(refined.cuts,'boundary_refinement')
            if improves(refined,best):
                best = refined
    except BudgetReached:
        exhausted = True
        before_refinement = locals().get('before_refinement',best)
        # Accepted intermediate moves remain admissible even at the budget limit.
        for candidate in fitter.partitions.values():
            if candidate is not None and improves(candidate,best):
                best = candidate
    assert not improves(baseline,best)
    return best, dict(seconds=perf_counter()-started,policy=PROPOSAL_POLICY,
                     candidate_family='production_path_retained' if best.cuts == baseline.cuts else 'direct_chain_partition',
                     selected_origin=origins.get(best.cuts,'boundary_refinement'),
                     selected_seed_origin=fitter.seed_origins.get(best.cuts, origins.get(best.cuts)),
                     raw_certificate_claim='none; qualified fusion reference is separate',selected_lambda=None,
                     baseline_cuts=baseline.cuts,baseline_score=baseline.score,
                     before_refinement_score=before_refinement.score,selected_cuts=best.cuts,
                     selected_score=best.score,selected_gap=best.gap,improved=improves(best,baseline),
                     proposal_budget_exhausted=exhausted,search_complete=not exhausted and not any(f['error'] == 'NumericalQualificationError' for f in fitter.failures),
                     completion_scope='bounded planned proposals only, never exhaustive partitions',
                     counters=fitter.counters,ward=work,refinements=refinements,failures=fitter.failures,
                     candidates=[dict(cuts=r.cuts,score=r.score,gap=r.gap,origin=origins.get(r.cuts,'boundary_refinement'))
                                 for r in fitter.partitions.values() if r is not None])
