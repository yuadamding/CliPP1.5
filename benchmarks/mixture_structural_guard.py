"""Conservative, truth-free selector for the soft-mixture development study.

Do not replace a baseline merely to rearrange a uniform-model partition at the
same K. Require lower new-criterion score than the best optimized uniform model
at baseline K, plus an occupied-cluster birth or an adaptive-model change.
Never reduce occupied K. This cannot guarantee accuracy; the joint gate remains.
"""
import math

import torch

from clipp1d.cuda.kernels import StructuralCompileBank
from clipp1d.cuda.mixture import MixtureResult, criterion, expectation


POLICY = dict(policy_id='preserve_partition_unless_structure_changes_v1',
              reference='best_uniform_soft_mixture_at_baseline_K',
              allow_uniform_same_K_reassignment=False, allow_occupied_K_decrease=False,
              require_em_fixed_point=True, score_margin='512*float64_eps*(1+abs(reference))',
              truth_used=False, cohort_used=False, global_optimality_certified=False)


@torch.no_grad()
def choose(model, records, policy, baseline_k):
    """Return a freshly checked candidate, or None to preserve ALL old estimates.

    Counts/CN, baseline K, frozen fit parameters and objective values are the only
    selection inputs. Baseline labels/centers/calls are not rewritten in this
    function. The caller binds their original artifact and copies them verbatim.
    """
    if isinstance(baseline_k, bool) or not isinstance(baseline_k, int) or baseline_k < 1:
        raise ValueError('Expected occupied baseline K')
    if model.device.type == 'cuda' and not model.kernels.compiled:
        raise ValueError('CUDA selection requires compiled tensor execution')
    estep = StructuralCompileBank(expectation) if model.device.type == 'cuda' else expectation
    checked = []
    with model.validated_stage():
        valid = torch.isfinite(model.log_prior)
        for index, record in enumerate(records):
            arrays = [torch.tensor(record[k], device=model.device, dtype=torch.float64)
                      for k in ('centers', 'weights', 'enrichment')]
            centers, weights, enrichment = arrays
            k = record['components']
            if (any(x.shape != (k,) or not bool(torch.isfinite(x).all()) for x in arrays) or
                    not bool((centers >= model.lower.max()).all() & (centers <= model.upper.min()).all() &
                             (weights >= 0).all() & ((weights.sum()-1).abs() < 1e-10) &
                             (enrichment >= 0).all() & (enrichment < 1).all())):
                raise ValueError('Invalid candidate parameter record')
            if not record['adaptive'] and bool((enrichment != 0).any()):
                raise ValueError('Uniform record contains nonzero enrichment')
            ll, post, _ = estep(model.alt, model.ref, model.slope, valid,
                                centers, weights, enrichment, model.eps)
            _, score = criterion(ll, enrichment, model.n, k, record['adaptive'],
                                  policy.enrichment_shrinkage)
            margin = 512*torch.finfo(torch.float64).eps*(1+abs(float(score)))
            if (not math.isfinite(record['score']) or
                    abs(float(score)-record['score']) > margin or
                    abs(float(ll)-record['log_likelihood']) > margin):
                raise ValueError('Candidate score does not match the original observations')
            occupied = int(torch.unique(post.sum(-1).argmax(-1)).numel())
            checked.append(dict(index=index, score=float(score), occupied=occupied))
        references = [v for v in checked if not records[v['index']]['adaptive'] and
                      records[v['index']]['components'] == baseline_k]
        if not references:
            raise ValueError('Missing optimized uniform reference at baseline K')
        reference = min(references, key=lambda v: (v['score'], v['index']))
        margin = 512*torch.finfo(torch.float64).eps*(1+abs(reference['score']))
        eligible = [v for v in checked if v['score'] < reference['score']-margin and
                    v['occupied'] >= baseline_k and
                    (v['occupied'] > baseline_k or records[v['index']]['adaptive']) and
                    records[v['index']]['status'] == 'em_fixed_point']
        decision = dict(policy=POLICY, baseline_k=baseline_k, reference=reference,
                         eligible=len(eligible), raw_certificate_inherited=False,
                         production_adopted=False)
        if not eligible:
            return None, dict(decision, selected_family='preserved_complete_graph_partition')
        selected = min(eligible, key=lambda v: (v['score'], records[v['index']]['components'],
                                               records[v['index']]['adaptive'], v['index']))
        r = records[selected['index']]
        centers, weights, enrichment = [torch.tensor(r[k], device=model.device, dtype=torch.float64)
                                        for k in ('centers', 'weights', 'enrichment')]
        ll, post, _ = estep(model.alt, model.ref, model.slope, valid,
                            centers, weights, enrichment, model.eps)
        result = MixtureResult(centers, weights, enrichment, post, float(ll), selected['score'],
                               r['adaptive'], r['status'], records, policy,
                               float(model.lower.max()), float(model.upper.min()),
                               'compiled_cuda' if model.device.type == 'cuda' else 'cpu_component_reference')
        return result, dict(decision, selected_family='new_mixture_structure', selected=selected)
