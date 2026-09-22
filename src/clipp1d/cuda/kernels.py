"""Pure tensor kernels. CUDA compilation is strict and never falls back to CPU.

The dense skew representation counts each unordered edge exactly once via 1/2
in energies and -row_sum in the adjoint. It avoids floating-point scatter atomics.
"""

from collections import OrderedDict
from types import FunctionType

import torch


def likelihood(alt, ref, slope, log_prior, phi, eps):
    # phi is [rows, proposals]; the final dimension always means multiplicity.
    mass = phi.unsqueeze(-1) * slope.unsqueeze(1)
    p = mass.clamp(eps, 1 - eps)
    joint = (
        alt[:, None, None] * p.log() + ref[:, None, None] * torch.log1p(-p) + log_prior[:, None, :]
    )
    norm = torch.logsumexp(joint, dim=-1)
    post = torch.exp(joint - norm.unsqueeze(-1))
    score = alt[:, None, None] / p - ref[:, None, None] / (1 - p)
    safe = torch.where(slope > 0, slope, torch.ones_like(slope))
    # Python scalar / Tensor lowers to reciprocal-then-multiply, whose double
    # rounding can miss an exact canonical NumPy kink by one ULP. Explicit
    # tensor/tensor division preserves the original represented breakpoint.
    low_kink = torch.full_like(safe, eps) / safe
    high_kink = torch.full_like(safe, 1 - eps) / safe
    at_low = (phi.unsqueeze(-1) == low_kink.unsqueeze(1)) & (slope > 0).unsqueeze(1)
    at_high = (phi.unsqueeze(-1) == high_kink.unsqueeze(1)) & (slope > 0).unsqueeze(1)
    # Preserve CountModel.evaluate's represented-mass derivative convention.
    # The separate one-sided audit additionally recognizes exact phi-space kinks.
    smooth = (mass > eps) & (mass < 1 - eps)
    inside = smooth & ~at_low & ~at_high
    moving = torch.where(smooth, slope[:, None, :], 0.0)
    grad = -(post * moving * score).sum(-1)
    curvature = (
        (
            post
            * moving.square()
            * (alt[:, None, None] / p.square() + ref[:, None, None] / (1 - p).square())
        )
        .sum(-1)
        .clamp_min(1e-8)
    )
    left = -(post * torch.where(inside | at_high, slope[:, None, :], 0.0) * score).sum(-1)
    right = -(post * torch.where(inside | at_low, slope[:, None, :], 0.0) * score).sum(-1)
    return -norm, grad, curvature, post, left, right


def differences(x):
    return x.unsqueeze(0) - x.unsqueeze(1)


def adjoint(q):
    # q[i,j] corresponds to x[j]-x[i]; q is skew-symmetric.
    return -q.sum(dim=-1)


def boxed_rank_one(h, b, lower, upper, rho):
    """Bounded diag(h+rho*N)-rho*11' solve by a scalar breakpoint scan.

    x_i(s)=clip((b_i+rho*s)/(h_i+rho*N),lower_i,upper_i), s=sum(x).
    Sorted entry/exit points give an affine equation on every interval. The
    denominator uses inactive counts and positive h/a terms, avoiding 1-near(1).
    Four safeguarded Newton polishing steps correct reduction roundoff. Caller
    independently audits the represented solution; no solve-then-clip shortcut.
    """
    n = h.numel()
    a = h + rho * n
    b = torch.where(lower < upper, b, a * lower)
    ratio = b / a
    events = torch.cat(((a * lower - b) / rho, (a * upper - b) / rho))
    order = torch.argsort(events, stable=True)
    events = events[order]
    dc = torch.cat((ratio - lower, upper - ratio))[order]
    count = torch.cat((torch.ones_like(h), -torch.ones_like(h)))[order].cumsum(0)
    fraction = torch.cat((h / a, -h / a))[order].cumsum(0)
    intercept = lower.sum() + dc.cumsum(0)
    denominator = (n - count + fraction) / n
    roots = torch.cat((lower.sum().reshape(1), intercept / denominator))
    lo_event = torch.cat((h.new_full((1,), -float("inf")), events))
    hi_event = torch.cat((events, h.new_full((1,), float("inf"))))
    slack = 128 * torch.finfo(h.dtype).eps * (1 + roots.abs())
    admissible = (roots >= lo_event - slack) & (roots <= hi_event + slack) & torch.isfinite(roots)
    selected = admissible.long().argmax().reshape(1)
    lo, hi = lower.sum(), upper.sum()
    s = roots.gather(0, selected)[0].clamp(lo, hi)
    for _ in range(4):
        unboxed = (b + rho * s) / a
        x = torch.maximum(lower, torch.minimum(upper, unboxed))
        f = s - x.sum()
        active = (unboxed > lower) & (unboxed < upper)
        derivative = ((~active).to(h.dtype).sum() + torch.where(active, h / a, 0.0).sum()) / n
        lo = torch.where(f < 0, s, lo)
        hi = torch.where(f > 0, s, hi)
        newton = s - f / derivative
        next_s = torch.where((newton >= lo) & (newton <= hi), newton, (lo + hi) * 0.5)
        s = torch.where(f == 0, s, next_s)
    return torch.maximum(lower, torch.minimum(upper, (b + rho * s) / a))


def edge_update(x, v, caps, rho):
    d = differences(x)
    a = d + v
    z = a.sign() * (a.abs() - caps / rho).clamp_min(0.0)
    v = a - z
    return z, v


def gap_kkt(x, q, h, target, lower, upper, caps):
    """Stable nonnegative-term primal-dual gap, without frozen constants."""
    a = adjoint(q)
    free = lower < upper
    safe_target = torch.where(free, target, x)
    g = torch.where(free, h * (x - safe_target), 0.0)
    unboxed = safe_target - a / h
    z = torch.maximum(lower, torch.minimum(upper, unboxed))
    delta = torch.where(free, x - z, 0.0)
    normal = torch.where(
        unboxed < lower,
        h * (lower - safe_target) + a,
        torch.where(unboxed > upper, h * (upper - safe_target) + a, 0.0),
    )
    node_normal = normal * delta
    d = differences(x)
    edge = caps * d.abs() - q * d
    margin_n = (
        128 * torch.finfo(x.dtype).eps * (1 + (h * (z - safe_target)).abs() + a.abs()) * delta.abs()
    )
    margin_e = 128 * torch.finfo(x.dtype).eps * (1 + caps) * d.abs()
    gap = (0.5 * h * delta.square() + node_normal.clamp_min(0.0)).sum() + 0.5 * edge.clamp_min(
        0.0
    ).sum()
    reference = torch.maximum(lower, torch.minimum(upper, safe_target))
    displacement = torch.where(free, x - reference, 0.0)
    energy = 0.5 * h * displacement.square() + h * (reference - safe_target) * displacement
    moving = free[:, None] | free[None, :]
    scale = energy.clamp_min(0.0).sum() + 0.5 * torch.where(moving, caps * d.abs(), 0.0).sum()
    r = g + a
    r = torch.where(x == lower, r.clamp_max(0.0), r)
    r = torch.where(x == upper, r.clamp_min(0.0), r)
    r = torch.where(free, r, 0.0)
    node = (r.abs() / (1 + g.abs() + a.abs())).max()
    dual = ((q - torch.maximum(-caps, torch.minimum(caps, q + d))).abs() / (1 + caps)).max()
    valid = (
        (x >= lower).all()
        & (x <= upper).all()
        & torch.isfinite(x).all()
        & torch.isfinite(q).all()
        & (q.abs() <= caps).all()
        & (node_normal >= -margin_n).all()
        & (edge >= -margin_e).all()
        & (q == -q.T).all()
        & torch.isfinite(gap)
        & torch.isfinite(scale)
        & torch.isfinite(node)
        & torch.isfinite(dual)
    )
    return torch.stack(
        (
            torch.where(valid, gap, gap.new_tensor(float("inf"))),
            torch.where(valid, scale, torch.zeros_like(scale)),
            torch.where(valid, torch.maximum(node, dual), gap.new_tensor(float("inf"))),
        )
    )


def compiler_probe(x):
    """A fail-closed admission probe detects globally disabled torch.compile."""
    return x + 1.0 if torch.compiler.is_compiling() else x - 1.0


class StructuralCompileBank:
    """Bounded compiled structural families with isolated Dynamo code objects.

    Dynamic extents alone do not cover singleton specialization, equal symbolic
    dimensions or input layout/alias guards. Each family fixes those structural
    facts while allowing numerical extents to vary. A local LRU bounds retained
    callables; global Dynamo cache limits are neither raised nor reset. Every
    miss still compiles strictly, and compilation errors propagate unchanged.
    """

    def __init__(self, function, *, backend="inductor", max_families=32):
        if isinstance(max_families, bool) or not isinstance(max_families, int) or max_families < 1:
            raise ValueError("Compile-family capacity must be a positive integer")
        self.function = function
        self.backend = backend
        self.max_families = max_families
        self._entries = OrderedDict()
        self.families_created = 0
        self.evictions = 0
        self.calls = 0

    @staticmethod
    def _normalize_and_key(arguments):
        normalized, objects, dimensions, aliases, object_aliases = [], {}, {}, {}, {}
        signature = []
        for value in arguments:
            if not isinstance(value, torch.Tensor):
                signature.append(("scalar", type(value).__name__, value))
                normalized.append(value)
                continue
            identity = id(value)
            if identity not in objects:
                array = value.contiguous()
                # Contiguity ignores singleton strides. Canonicalize those too,
                # otherwise size-one views consume separate compiler guards.
                stride = 1
                strides = []
                for size in reversed(array.shape):
                    strides.append(stride)
                    stride *= max(1, size)
                canonical = tuple(reversed(strides))
                if array.stride() != canonical:
                    array = array.as_strided(array.shape, canonical)
                objects[identity] = array
            array = objects[identity]
            normalized.append(array)
            shape = []
            for size in array.shape:
                if size <= 1:
                    shape.append(("constant", int(size)))
                else:
                    # Equality relations matter to Dynamo's symbolic duck-sizing;
                    # actual extents do not belong in the family cache key.
                    shape.append(("symbol", dimensions.setdefault(int(size), len(dimensions))))
            storage = (array.device, array.untyped_storage().data_ptr())
            alias = aliases.setdefault(storage, len(aliases))
            object_alias = object_aliases.setdefault(id(array), len(object_aliases))
            signature.append(
                (
                    "tensor",
                    str(array.dtype),
                    str(array.device),
                    tuple(shape),
                    alias,
                    object_alias,
                    bool(array.requires_grad),
                )
            )
        return tuple(normalized), (torch.is_grad_enabled(), tuple(signature))

    def __call__(self, *arguments):
        normalized, key = self._normalize_and_key(arguments)
        compiled = self._entries.get(key)
        if compiled is None:
            serial = self.families_created
            # Function closures alone still share __code__ and its global Dynamo
            # specialization budget. Distinct code objects isolate each declared
            # family without changing the function's bytecode or mathematics.
            name = f"{self.function.__name__}__bank_{id(self)}_family_{serial}"
            replacements = {"co_name": name}
            if hasattr(self.function.__code__, "co_qualname"):
                replacements["co_qualname"] = name
            code = self.function.__code__.replace(**replacements)
            isolated = FunctionType(
                code,
                self.function.__globals__,
                name,
                self.function.__defaults__,
                self.function.__closure__,
            )
            compiled = torch.compile(isolated, backend=self.backend, fullgraph=True, dynamic=True)
            self._entries[key] = compiled
            self.families_created += 1
            if len(self._entries) > self.max_families:
                self._entries.popitem(last=False)
                self.evictions += 1
        else:
            self._entries.move_to_end(key)
        self.calls += 1
        return compiled(*normalized)

    def diagnostics(self):
        return dict(
            policy="bounded_structural_families_v1",
            backend=self.backend if isinstance(self.backend, str) else "test_reference_backend",
            fullgraph=True,
            max_families=self.max_families,
            active_families=len(self._entries),
            families_created=self.families_created,
            evictions=self.evictions,
            calls=self.calls,
            dynamic_extents=True,
            global_cache_limit_modified=False,
            eager_fallback=False,
        )


class Kernels:
    """A small per-fit compilation bank; no catch-and-ignore compiler failure."""

    def __init__(self, device, compiled=True):
        self.compiled = compiled and torch.device(device).type == "cuda"
        if self.compiled:
            if torch._dynamo.config.suppress_errors:
                raise RuntimeError(
                    "Disable torch._dynamo.config.suppress_errors for qualified CUDA inference"
                )
            probe = torch.compile(compiler_probe, backend="inductor", fullgraph=True)
            if not bool(probe(torch.zeros((), device=device, dtype=torch.float64)) == 1.0):
                raise RuntimeError("CUDA compilation was disabled; compiled-path admission failed")
        for name in ("likelihood", "boxed_rank_one", "edge_update", "gap_kkt"):
            fn = globals()[name]
            if self.compiled:
                fn = StructuralCompileBank(fn)
            setattr(self, name, fn)

    def compilation_diagnostics(self):
        return {
            name: getattr(self, name).diagnostics()
            for name in ("likelihood", "boxed_rank_one", "edge_update", "gap_kkt")
            if isinstance(getattr(self, name), StructuralCompileBank)
        }
