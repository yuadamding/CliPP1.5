"""GPU memberships and independent, unconstrained secondary likelihood refits."""
from dataclasses import dataclass
import torch
from .scalar import Problems, ScalarBatch, solve_scalar
from .policy import CudaPolicy, QualificationError
from .kernels import differences, adjoint


def grouping(x, tolerance, *, separate_clonal=True):
    """Bounded-diameter groups, canonically labeled by their minimum node index.

    Current-value order only determines tolerance runs. An overwide connected
    tolerance run keeps only its exact-value groups. This conservative parallel
    rule bounds diameter without breaking exact fusions. Exact-one nodes never
    group with merely near-one nodes.
    """
    if x.ndim != 1 or x.numel() == 0 or not 0 <= tolerance < float("inf"):
        raise ValueError("Grouping requires a nonempty vector and finite nonnegative tolerance")
    if x.dtype != torch.float64 or not bool(torch.isfinite(x).all()):
        raise ValueError("Grouping requires finite float64 CCFs")
    n = x.numel()
    value_order = torch.argsort(x, stable=True)
    values = x[value_order]
    idx = torch.arange(n, device=x.device)
    cuts = torch.ones(n, device=x.device, dtype=torch.bool)
    cuts[1:] = values[1:] - values[:-1] > tolerance
    if separate_clonal:
        cuts[1:] |= (values[1:] == 1) != (values[:-1] == 1)
    start = torch.cummax(torch.where(cuts, idx, 0), 0).values
    stops = torch.ones_like(cuts)
    stops[:-1] = cuts[1:]
    end = torch.cummin(torch.where(stops, idx, n - 1).flip(0), 0).values.flip(0)
    exact_boundary = torch.ones_like(cuts)
    exact_boundary[1:] = values[1:] != values[:-1]
    cuts |= (values[end] - values[start] > tolerance) & exact_boundary
    value_labels = cuts.long().cumsum(0) - 1
    first_node = torch.full((n,), n, device=x.device, dtype=torch.long)
    first_node.scatter_reduce_(0, value_labels, value_order, reduce="amin")
    block_order = torch.argsort(first_node, stable=True)
    canonical = torch.empty_like(block_order)
    canonical[block_order] = idx
    labels = torch.empty_like(value_labels)
    labels[value_order] = canonical[value_labels]
    order = torch.argsort(labels, stable=True)
    sorted_labels = labels[order]
    counts = torch.zeros(n, dtype=torch.long, device=x.device)
    counts.scatter_add_(0, labels, torch.ones_like(labels))
    return order, sorted_labels, labels, counts


def polish_quadratic(x, q, h, target, lower, upper, caps, tolerance):
    """A numerical equality proposal only; caller must re-audit the original QP."""
    order, _, labels, lengths = grouping(x, tolerance, separate_clonal=False)
    same = labels[:, None] == labels[None, :]
    external = adjoint(torch.where(same, 0., caps * differences(x).sign()))

    def reduce(v, op="sum"):
        return torch.segment_reduce(v[order], op, lengths=lengths)

    total_h = reduce(h)
    safe_target = torch.where(lower < upper, target, lower)
    center = reduce(h * safe_target - external) / total_h.clamp_min(torch.finfo(x.dtype).tiny)
    lo, hi = reduce(lower, "max"), reduce(upper, "min")
    center = torch.maximum(lo, torch.minimum(hi, center))
    candidate = center[labels]
    return torch.where(torch.isfinite(candidate) & (lo <= hi)[labels], candidate, x)


@dataclass
class Refit:
    labels: torch.Tensor
    centers: torch.Tensor
    phi: torch.Tensor
    sizes: torch.Tensor
    clonal: int  # Closest-to-one designation only; no center is fixed at one.
    loss: torch.Tensor
    gap: torch.Tensor
    score: torch.Tensor
    scalar: ScalarBatch


_SCALAR_FIELDS = ("phi", "loss", "lower_bound", "gap", "alternative", "qualified")


def _clone_scalar(scalar):
    return ScalarBatch(*(getattr(scalar, name).clone() for name in _SCALAR_FIELDS),
                       subdivisions=scalar.subdivisions)


def _clone_refit(result):
    return Refit(*(getattr(result, name).clone() for name in
                   ("labels", "centers", "phi", "sizes")), result.clonal,
                 result.loss.clone(), result.gap.clone(), result.score.clone(),
                 _clone_scalar(result.scalar))


class _Snapshot:
    """Small boundary-only guard, including mutations made through Tensor.data."""

    def __init__(self, tensors):
        self.versions = tuple((id(v), v._version) for v in tensors)
        self.values = tuple(v.detach().clone() for v in tensors)

    def validate(self, tensors, name):
        if (tuple((id(v), v._version) for v in tensors) != self.versions or
                not bool(torch.stack([(v == frozen).all()
                                      for v, frozen in zip(tensors, self.values)]).all())):
            raise ValueError(f"{name} tensors were modified")


def _qualified_scalar(scalar, lower, upper, policy):
    tensors = tuple(getattr(scalar, name) for name in _SCALAR_FIELDS)
    if (any(v.shape != lower.shape or v.device != lower.device for v in tensors) or
            any(v.dtype != torch.float64 for v in tensors[:-1]) or
            scalar.qualified.dtype != torch.bool):
        raise ValueError("Scalar refit fields must match canonical membership dimensions")
    return (scalar.qualified.all() & torch.isfinite(scalar.phi).all() &
            torch.isfinite(scalar.loss).all() & torch.isfinite(scalar.lower_bound).all() &
            torch.isfinite(scalar.gap).all() & (scalar.gap >= 0).all() &
            torch.isfinite(scalar.alternative).all() &
            (scalar.phi >= lower).all() & (scalar.phi <= upper).all() &
            (scalar.gap == (scalar.loss - scalar.lower_bound).clamp_min(0.)).all() &
            (scalar.gap <= policy.scalar_atol + policy.scalar_rtol * scalar.loss.abs()).all())


class QualifiedPilot:
    """Private qualified copy tied to this model's canonical mutation identities.

    Construct immediately after computing the model's pilot. Matching array
    positions in another model, subset, or policy cannot authorize reuse.
    """

    def __init__(self, model, scalar, policy=CudaPolicy()):
        model.validate(full=True)
        if not bool(_qualified_scalar(scalar, model.lower, model.upper, policy)):
            raise QualificationError("Singleton pilot reuse requires qualified scalar results")
        loss = model.loss(scalar.phi)
        margin = 128 * torch.finfo(torch.float64).eps * (1 + loss.abs())
        if not bool(((loss - scalar.loss).abs() <= margin).all()):
            raise QualificationError("Singleton pilot losses do not match the canonical model")
        self.model, self.policy = model, policy
        self.mutation_ids = tuple(model.mutation_ids)
        self.scalar = _clone_scalar(scalar)
        self._snapshot = _Snapshot(self._tensors())
        self._subdivisions = self.scalar.subdivisions

    def _tensors(self):
        return tuple(getattr(self.scalar, name) for name in _SCALAR_FIELDS)

    def validate(self, model, policy):
        if (model is not self.model or policy != self.policy or
                tuple(model.mutation_ids) != self.mutation_ids):
            raise ValueError("Singleton pilot reuse is bound to its canonical model and policy")
        model.validate(full=True)
        self._snapshot.validate(self._tensors(), "Qualified pilot")
        if self.scalar.subdivisions != self._subdivisions:
            raise ValueError("Qualified pilot metadata was modified")


class LastPartitionCache:
    """One private qualified membership refit, never an unbounded path history."""

    def __init__(self, model, policy=CudaPolicy()):
        model.validate(full=True)
        self.model, self.policy = model, policy
        self.mutation_ids = tuple(model.mutation_ids)
        self.result = None
        self.refits_computed = self.refits_reused = self.singleton_pilots_reused = 0

    def _tensors(self):
        r = self.result
        return tuple(getattr(r, name) for name in
                     ("labels", "centers", "phi", "sizes", "loss", "gap", "score")) + tuple(
                         getattr(r.scalar, name) for name in _SCALAR_FIELDS)

    def get(self, model, labels, policy, *, allow_reuse=True):
        if (model is not self.model or policy != self.policy or
                tuple(model.mutation_ids) != self.mutation_ids):
            raise ValueError("Partition refit cache is bound to its canonical model and policy")
        model.validate(full=True)
        if self.result is None:
            return None
        self._snapshot.validate(self._tensors(), "Cached refit")
        if (self.result.clonal, self.result.scalar.subdivisions) != self._metadata:
            raise ValueError("Cached refit metadata was modified")
        if not allow_reuse:
            return None
        if not torch.equal(labels, self.result.labels):
            return None
        self.refits_reused += 1
        return _clone_refit(self.result)

    def _store_qualified(self, result):
        self.result = _clone_refit(result)
        self._snapshot = _Snapshot(self._tensors())
        self._metadata = self.result.clonal, self.result.scalar.subdivisions


def partition_score(loss, sizes):
    if (sizes.ndim != 1 or sizes.numel() == 0 or
            not bool(torch.isfinite(sizes).all() & (sizes > 0).all() &
                     (sizes == sizes.round()).all() & torch.isfinite(loss))):
        raise ValueError("Score requires finite loss and occupied integer memberships")
    k = sizes.numel()
    n = sizes.sum().to(torch.float64)
    s = sizes.to(torch.float64)
    kt = n.new_tensor(float(k))
    mass = torch.lgamma(kt) - torch.lgamma(n + kt) + torch.lgamma(s + 1).sum() + torch.lgamma(kt + 1)
    score = 2 * loss + kt * n.log() - 1.4 * mass
    if not bool(torch.isfinite(score)):
        raise QualificationError("Partition score is nonfinite")
    return score


def canonical_labels(labels):
    """Relabel occupied groups by first node, without changing any membership."""
    if labels.ndim != 1 or labels.numel() == 0 or labels.dtype != torch.long:
        raise ValueError("Memberships must be a nonempty int64 vector")
    _, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    n = labels.numel()
    counts = torch.bincount(inverse)
    first = torch.full_like(counts, n)
    first.scatter_reduce_(0, inverse, torch.arange(n, device=labels.device), reduce="amin")
    blocks = torch.argsort(first, stable=True)
    rename = torch.empty_like(blocks)
    rename[blocks] = torch.arange(blocks.numel(), device=labels.device)
    result = rename[inverse]
    return result, torch.argsort(result, stable=True), counts[blocks]


def refit(model, x, policy=CudaPolicy(), *, pilot_reuse=None, cache=None,
          separate_exact_one=True):
    """Extract raw memberships and delegate to the explicit-label refitter.

    Legacy raw-path extraction retains exact-one separation for a controlled
    baseline. Generic extraction is an explicit, separately measured variant.
    """
    if x.shape != model.lower.shape or x.device != model.device:
        raise ValueError("Refit CCF vector must match the model")
    labels = grouping(x, policy.fusion_tol, separate_clonal=separate_exact_one)[2]
    return refit_labels(model, labels, policy, pilot_reuse=pilot_reuse, cache=cache)


def refit_labels(model, labels, policy=CudaPolicy(), *, incumbent_centers=None,
                 pilot_reuse=None, cache=None):
    """Qualify the supplied memberships; equal centers never merge groups.

    Incumbent centers, when supplied, follow canonical first-node group order,
    independent of the caller's integer label names. They are feasible loss
    witnesses, not inherited scalar certificates.
    """
    with model.validated_stage():
        return _refit_labels(model, labels, policy, incumbent_centers=incumbent_centers,
                             pilot_reuse=pilot_reuse, cache=cache)


def _refit_labels(model, labels, policy, *, incumbent_centers, pilot_reuse, cache):
    if labels.shape != model.lower.shape or labels.device != model.device:
        raise ValueError("Refit memberships must match the model")
    model.validate(full=True)
    labels, order, lengths = canonical_labels(labels)
    k = lengths.numel()
    lower = torch.segment_reduce(model.lower[order], "max", lengths=lengths)
    upper = torch.segment_reduce(model.upper[order], "min", lengths=lengths)
    if not bool((lower <= upper).all()):
        raise QualificationError("Membership proposal has an empty feasible interval")
    if incumbent_centers is not None:
        c = incumbent_centers
        if (c.shape != (k,) or c.dtype != torch.float64 or c.device != model.device or
                not bool(torch.isfinite(c).all() & (c >= lower).all() & (c <= upper).all())):
            raise ValueError("Incumbent centers must be feasible float64 canonical centers")
    if cache is not None:
        cached = cache.get(model, labels, policy, allow_reuse=incumbent_centers is None)
        # An incumbent can supply a better feasible point than a cached rounded
        # scalar solution. Do not silently discard that witness.
        if cached is not None:
            return cached
    singleton = lengths == 1
    reused = int(singleton.sum()) if pilot_reuse is not None else 0
    if pilot_reuse is not None:
        if not isinstance(pilot_reuse, QualifiedPilot):
            raise ValueError("Singleton reuse requires a model-bound QualifiedPilot")
        pilot_reuse.validate(model, policy)
    if cache is not None:
        # Count attempted new refits, including those later left unresolved.
        # Only qualified results are eligible for the one retained cache entry.
        cache.refits_computed += 1
        cache.singleton_pilots_reused += reused
    if reused:
        # The first ordered node identifies each canonical membership, including
        # noncontiguous groups. A group's label is not a pilot row index.
        starts = lengths.cumsum(0) - lengths
        nodes = order[starts[singleton]]
        fields = [torch.empty(k, device=model.device, dtype=getattr(pilot_reuse.scalar, name).dtype)
                  for name in _SCALAR_FIELDS]
        for out, name in zip(fields, _SCALAR_FIELDS):
            out[singleton] = getattr(pilot_reuse.scalar, name)[nodes]
        subdivisions = 0
        if reused < k:
            unresolved_order = order[~singleton[labels[order]]]
            remaining = solve_scalar(Problems(model.subset(unresolved_order), lengths[~singleton]), policy)
            for out, name in zip(fields, _SCALAR_FIELDS):
                out[~singleton] = getattr(remaining, name)
            subdivisions = remaining.subdivisions
        scalar = ScalarBatch(*fields, subdivisions=subdivisions)
    else:
        scalar = solve_scalar(Problems(model.subset(order), lengths), policy)
    if incumbent_centers is not None:
        losses = model.loss(incumbent_centers[labels])
        proposed = torch.segment_reduce(losses[order], "sum", lengths=lengths)
        improve = proposed < scalar.loss
        value = torch.where(improve, proposed, scalar.loss)
        bound = torch.minimum(scalar.lower_bound, value)
        gap = (value - bound).clamp_min(0.)
        scalar = ScalarBatch(torch.where(improve, incumbent_centers, scalar.phi), value,
                             bound, gap, scalar.alternative,
                             torch.isfinite(bound) & torch.isfinite(gap) &
                             (gap <= policy.scalar_atol + policy.scalar_rtol * value.abs()),
                             scalar.subdivisions)
    valid = _qualified_scalar(scalar, lower, upper, policy)
    if not bool(valid):
        raise QualificationError("Secondary membership refit unresolved", maximum_gap=float(scalar.gap.max()))
    centers = scalar.phi.clone()
    clonal = int((centers - 1.).abs().argmin())
    loss, gap = scalar.loss.sum(), scalar.gap.sum()
    if not bool(torch.isfinite(loss) & torch.isfinite(gap)):
        raise QualificationError("Aggregate membership loss or scalar gap is nonfinite")
    result = Refit(labels, centers, centers[labels], lengths, clonal, loss, gap,
                   partition_score(loss, lengths), scalar)
    if cache is not None:
        cache._store_qualified(result)
    return result
