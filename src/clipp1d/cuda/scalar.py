"""Batched device-side interval search for pilots and arbitrary membership refits.

Host control reads one batch qualification flag, never per-mutation likelihood
arrays. Interval budgets and float64 admission tolerances match the source policy.
Bounds are numerical float64 qualifications, not directed-rounding proofs.
"""
from dataclasses import dataclass
import torch
from .model import TensorModel
from .policy import CudaPolicy, QualificationError


@dataclass
class ScalarBatch:
    phi: torch.Tensor
    loss: torch.Tensor
    lower_bound: torch.Tensor
    gap: torch.Tensor
    alternative: torch.Tensor
    qualified: torch.Tensor
    subdivisions: int


class Problems:
    """Rows sorted into groups; no dense group-by-mutation membership matrix."""

    def __init__(self, model: TensorModel, lengths: torch.Tensor):
        model.validate()
        if (lengths.ndim != 1 or lengths.numel() == 0 or lengths.dtype != torch.long or
                lengths.device != model.device or
                not bool((lengths > 0).all() & (lengths.sum() == model.n))):
            raise ValueError("Scalar group lengths must be positive device integers covering all rows")
        self.model = model
        self.lengths = lengths.clone()
        self.count = lengths.numel()
        self.group = torch.repeat_interleave(torch.arange(self.count, device=model.device),
                                            lengths, output_size=model.n)
        self.lower = self.reduce(model.lower, "max")
        self.upper = self.reduce(model.upper, "min")
        if not bool((self.lower <= self.upper).all()):
            raise ValueError("Scalar group has no common feasible domain")
        self._versions = tuple((id(v), v._version) for v in
                               (self.lengths, self.group, self.lower, self.upper))

    def validate(self):
        self.model.validate()
        if tuple((id(v), v._version) for v in
                 (self.lengths, self.group, self.lower, self.upper)) != self._versions:
            raise ValueError("Scalar membership or domain tensors were modified")

    def reduce(self, x, reduction="sum"):
        return torch.segment_reduce(x, reduction, lengths=self.lengths)

    def evaluate(self, points):
        self.validate()
        if (points.ndim != 2 or points.shape[0] != self.count or
                points.dtype != torch.float64 or points.device != self.model.device):
            raise ValueError("Scalar proposal points must be a matching float64 device matrix")
        row_phi = points[self.group]
        out = self.model.kernels.likelihood(self.model.alt, self.model.ref, self.model.slope,
                                            self.model.log_prior, row_phi, self.model.eps)
        return self.reduce(out[0]), self.reduce(out[1])

    def bounds(self, left, right):
        self.validate()
        m = self.model
        lo, r = left[self.group], right[self.group]
        sl = m.slope[:, None, :]
        pl = (sl * lo[..., None]).clamp(m.eps, 1 - m.eps)
        pr = (sl * r[..., None]).clamp(m.eps, 1 - m.eps)
        empirical = (m.alt / (m.alt + m.ref))[:, None, None]
        best = torch.maximum(pl, torch.minimum(pr, empirical))
        logits = (m.alt[:, None, None] * best.log()
                  + m.ref[:, None, None] * torch.log1p(-best) + m.log_prior[:, None, :])
        independent = self.reduce(-torch.logsumexp(logits, -1))
        mid = left + (right - left) * .5
        loss, gradient = self.evaluate(mid)
        safe = torch.where(m.slope > 0, m.slope, torch.ones_like(m.slope))
        # Tensor/Tensor true division preserves canonical float64 clipping
        # points. Python scalar/Tensor division uses reciprocal multiplication,
        # which can move an exact boundary by one ULP (for example slope .4).
        lk = (torch.full_like(safe, m.eps) / safe)[:, None, :]
        hk = (torch.full_like(safe, 1 - m.eps) / safe)[:, None, :]
        valid = torch.isfinite(m.log_prior)[:, None, :]
        # A nominal kink at either endpoint is also nonsmooth for float64
        # arithmetic: slope * (threshold / slope) can round one ULP inside the
        # unclipped region. The independent candidate bound remains valid.
        crossing = (((lk >= lo[..., None]) & (lk <= r[..., None]))
                    | ((hk >= lo[..., None]) & (hk <= r[..., None]))) & valid
        crossing = self.reduce(crossing.any(-1).to(m.alt.dtype), "max") > 0
        mass = sl * mid[self.group][..., None]
        moving = torch.where((mass > m.eps) & (mass < 1 - m.eps), sl, 0.)
        score_l = moving * (m.alt[:, None, None] / pl - m.ref[:, None, None] / (1 - pl))
        score_r = moving * (m.alt[:, None, None] / pr - m.ref[:, None, None] / (1 - pr))
        spread = (torch.where(valid, score_l, -float("inf")).amax(-1)
                  - torch.where(valid, score_r, float("inf")).amin(-1))
        spread = torch.where(valid.sum(-1) == 1, 0., spread)
        negative_curvature = self.reduce(spread.square() / 4)
        # The represented midpoint can equal an endpoint on one-ULP intervals.
        radius = torch.maximum(mid - left, right - mid)
        smooth = loss - gradient.abs() * radius - .5 * negative_curvature * radius.square()
        # Rounded slope*phi affects log(p) and log(1-p) most near clipping.
        # A loss-relative epsilon margin alone misses that conditioning (one
        # probability ULP can change ref*log(1-p) by ~1e-8 at p=1-eps). Bound
        # this input-rounding contribution before using the Taylor envelope.
        # The independent candidate envelope remains available without this
        # deduction, so numerical admission tolerances do not change.
        sensitivity = (m.alt[:, None, None] / pl + m.ref[:, None, None] / (1 - pr))
        sensitivity = torch.where(valid, sensitivity, 0.).amax(-1)
        smooth -= 32 * torch.finfo(torch.float64).eps * self.reduce(sensitivity)
        bound = torch.where(crossing, independent, torch.maximum(independent, smooth))
        margin = 128 * torch.finfo(torch.float64).eps * (1 + bound.abs() + loss.abs())
        return bound - margin, mid, loss

    def nearest_kink(self, left, right):
        self.validate()
        m = self.model
        safe = torch.where(m.slope > 0, m.slope, torch.ones_like(m.slope))
        kinks = torch.cat((torch.full_like(safe, m.eps) / safe,
                           torch.full_like(safe, 1 - m.eps) / safe), -1)
        valid = torch.isfinite(m.log_prior).repeat(1, 2)
        lo, r = left[self.group], right[self.group]
        mid = lo + (r - lo) * .5
        inside = valid & (kinks > lo[:, None]) & (kinks < r[:, None])
        distance = torch.where(inside, (kinks - mid[:, None]).abs(), float("inf"))
        closest = self.reduce(distance.amin(-1), "min")
        point = self.reduce(torch.where(inside & (distance == closest[self.group, None]),
                                        kinks, float("inf")).amin(-1), "min")
        return torch.where(torch.isfinite(point), point, left + (right - left) * .5)


def solve_scalar(problems: Problems, policy=CudaPolicy()):
    p, m, b = problems, problems.model, problems.count
    p.validate()
    if b > policy.scalar_batch_size:
        # Only batch-boundary counts enter host control; no likelihood, fitted
        # coordinate, or optimization array is transferred to or solved on CPU.
        results = []
        row_start = 0
        for group_start in range(0, b, policy.scalar_batch_size):
            lengths = p.lengths[group_start:group_start + policy.scalar_batch_size]
            row_stop = row_start + int(lengths.sum())
            batch = Problems(m.subset(slice(row_start, row_stop)), lengths)
            results.append(solve_scalar(batch, policy))
            row_start = row_stop
        return ScalarBatch(*(torch.cat([getattr(result, key) for result in results]) for key in
                             ("phi", "loss", "lower_bound", "gap", "alternative", "qualified")),
                           subdivisions=sum(result.subdivisions for result in results))
    device = m.device
    grid = torch.linspace(0., 1., 33, device=device, dtype=torch.float64)
    seeds = p.lower[:, None] + (p.upper - p.lower)[:, None] * grid
    # Candidate-mode averages are attained seeds only; bounds still cover the full domain.
    modes = ((m.alt / (m.alt + m.ref))[:, None]
             / torch.where(m.slope > 0, m.slope, torch.ones_like(m.slope)))
    valid = torch.isfinite(m.log_prior)
    modes = p.reduce(torch.where(valid, modes, 0.)) / p.reduce(valid.to(torch.float64)).clamp_min(1.)
    modes = torch.maximum(p.lower[:, None], torch.minimum(p.upper[:, None], modes))
    seeds = torch.sort(torch.cat((seeds, modes), -1), dim=-1).values
    values, _ = p.evaluate(seeds)
    best_loss, idx = values.min(-1)
    best_x = seeds.gather(1, idx[:, None])[:, 0]
    # Exact same-slope single-candidate groups (including zero-alt clipping plateaus).
    single = p.reduce((valid.sum(-1) == 1).to(torch.float64), "min") == 1
    # The unique valid support need not occupy storage column zero.
    candidate_slope = torch.where(valid, m.slope, 0.).sum(-1)
    smin, smax = p.reduce(candidate_slope, "min"), p.reduce(candidate_slope, "max")
    exact = single & (smin == smax)
    alt, ref = p.reduce(m.alt), p.reduce(m.ref)
    empirical = alt / (alt + ref)
    plo, phi = (smin * p.lower).clamp(m.eps, 1 - m.eps), (smin * p.upper).clamp(m.eps, 1 - m.eps)
    direct_x = (empirical.clamp_max(1 - m.eps) / smin).clamp(p.lower, p.upper)
    direct_x = torch.where((empirical <= plo) | (plo == phi), p.lower, direct_x)
    direct_loss = p.evaluate(direct_x[:, None])[0][:, 0]
    exact_lb = direct_loss - 128 * torch.finfo(torch.float64).eps * (1 + direct_loss.abs())
    best_x = torch.where(exact, direct_x, best_x)
    best_loss = torch.where(exact, direct_loss, best_loss)
    if bool(exact.all()):
        gap = (best_loss - exact_lb).clamp_min(0.)
        qualified = (torch.isfinite(best_x) & torch.isfinite(best_loss) & torch.isfinite(exact_lb) &
                     (best_x >= p.lower) & (best_x <= p.upper) &
                     (gap <= policy.scalar_atol + policy.scalar_rtol * best_loss.abs()))
        return ScalarBatch(best_x, best_loss, exact_lb, gap, best_x.clone(), qualified, 0)
    # Local golden searches produce multistart proposals, never scalar certificates.
    if not bool(exact.all()):
        local = (values[:, 1:-1] <= values[:, :-2]) & (values[:, 1:-1] <= values[:, 2:])
        a, z = seeds[:, :-2].clone(), seeds[:, 2:].clone()
        ratio = (5. ** .5 - 1) / 2
        for _ in range(48):
            c, d = z - ratio * (z - a), a + ratio * (z - a)
            fc, _ = p.evaluate(c)
            fd, _ = p.evaluate(d)
            take = fc <= fd
            a, z = torch.where(take, a, c), torch.where(take, d, z)
        wells = (a + z) * .5
        well_loss, _ = p.evaluate(wells)
        well_loss = torch.where(local, well_loss, float("inf"))
        win, wi = well_loss.min(-1)
        wx = wells.gather(1, wi[:, None])[:, 0]
        improve = (~exact) & (win < best_loss)
        best_x, best_loss = torch.where(improve, wx, best_x), torch.where(improve, win, best_loss)
    else:
        wells, well_loss = seeds, values
    initial = seeds.shape[1] - 1
    capacity = initial + policy.scalar_max_intervals
    left = torch.zeros((b, capacity), dtype=torch.float64, device=device)
    right = torch.zeros_like(left)
    bounds = torch.full_like(left, float("inf"))
    left[:, :initial], right[:, :initial] = seeds[:, :-1], seeds[:, 1:]
    bounds[:, :initial] = p.bounds(left[:, :initial], right[:, :initial])[0]
    slots, rounds = initial, 0
    row = torch.arange(b, device=device)
    for iteration in range(policy.scalar_max_intervals + 1):
        smallest, at = bounds[:, :slots].min(-1)
        lb = torch.where(exact, exact_lb, torch.minimum(best_loss, smallest))
        gap = (best_loss - lb).clamp_min(0.)
        qualified = (torch.isfinite(best_x) & torch.isfinite(best_loss) & torch.isfinite(lb) &
                     torch.isfinite(gap) & (best_x >= p.lower) & (best_x <= p.upper) &
                     (gap <= policy.scalar_atol + policy.scalar_rtol * best_loss.abs()))
        if iteration % policy.check_every == 0 and bool(qualified.all()):
            break
        if iteration == policy.scalar_max_intervals:
            break
        lo, r = left[row, at], right[row, at]
        cut = p.nearest_kink(lo, r)
        active = (~qualified) & (cut > lo) & (cut < r)
        # Inactive lanes retain their whole-interval lower bound; no false qualification.
        lnew, rnew = torch.stack((lo, cut), -1), torch.stack((cut, r), -1)
        lbnew, mid, midloss = p.bounds(lnew, rnew)
        cutloss = p.evaluate(cut[:, None])[0][:, 0]
        props, prop_loss = torch.cat((cut[:, None], mid), -1), torch.cat((cutloss[:, None], midloss), -1)
        v, atp = prop_loss.min(-1)
        t = props.gather(1, atp[:, None])[:, 0]
        improve = active & ((v < best_loss) | ((v == best_loss) & (t < best_x)))
        best_x, best_loss = torch.where(improve, t, best_x), torch.where(improve, v, best_loss)
        left[row, at] = torch.where(active, lnew[:, 0], lo)
        right[row, at] = torch.where(active, rnew[:, 0], r)
        bounds[row, at] = torch.where(active, lbnew[:, 0], bounds[row, at])
        left[:, slots], right[:, slots] = lnew[:, 1], rnew[:, 1]
        bounds[:, slots] = torch.where(active, lbnew[:, 1], float("inf"))
        slots += 1
        rounds += 1
    delta = ((p.upper - p.lower) * 1e-4).clamp_max(1e-5)
    wl, _ = p.evaluate(torch.maximum(p.lower[:, None], wells - delta[:, None]))
    wr, _ = p.evaluate(torch.minimum(p.upper[:, None], wells + delta[:, None]))
    admissible = ((wells - best_x[:, None]).abs() > 1e-3) & (well_loss <= best_loss[:, None] + 2.)
    admissible &= (well_loss <= wl) & (well_loss <= wr)
    alt_loss, ai = torch.where(admissible, well_loss, float("inf")).min(-1)
    alternative = torch.where(torch.isfinite(alt_loss), wells.gather(1, ai[:, None])[:, 0], best_x)
    return ScalarBatch(best_x, best_loss, lb, gap, alternative, qualified, rounds)


def pilot(model: TensorModel, policy=CudaPolicy()):
    results = []
    for start in range(0, model.n, policy.scalar_batch_size):
        stop = min(model.n, start + policy.scalar_batch_size)
        sub = model.subset(slice(start, stop))
        lengths = torch.ones(stop - start, dtype=torch.long, device=model.device)
        result = solve_scalar(Problems(sub, lengths), policy)
        if not bool(result.qualified.all()):
            raise QualificationError("CUDA scalar pilot unresolved", batch_start=start,
                                      maximum_gap=float(result.gap.max()))
        results.append(result)
    return ScalarBatch(*(torch.cat([getattr(r, key) for r in results]) for key in
                         ("phi", "loss", "lower_bound", "gap", "alternative", "qualified")),
                        subdivisions=sum(r.subdivisions for r in results))
