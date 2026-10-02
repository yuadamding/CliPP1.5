# Normative estimator contract

This describes the numerical estimator preserved from commit
`15d397b680ed3392a2ec3bbead1e5d5b460448ff`. Engineering version 1.5.1 changes input,
build and publication contracts, not the model/search/scoring identities.
Machine-readable identities and numerical controls are in `clipp.versions`.

For purity p, alternate count r_i, total reads n_i, major CN M_i and total CN C_i,
the cellular prevalence c lies in [0,p]. Cancer cell fraction is x=c/p. Normal CN
is 2. The mutation likelihood includes the binomial coefficient:

\[
L_i(c)=\frac{1}{M_i}\sum_{m=1}^{M_i}
 {n_i\choose r_i}
 \left(\frac{c m}{2(1-p)+p C_i}\right)^{r_i}
 \left(1-\frac{c m}{2(1-p)+p C_i}\right)^{n_i-r_i}.
\]

Multiplicity support is **all** integers 1..M_i, including equal major/minor CN
states. No rounded VAF call, truth-derived state, alternate prior or clonal
constraint enters fitting. Final conditional multiplicity probabilities normalize
these component likelihoods at each mutation's assigned final CP.

## Initialization and frozen representation

Identical count/CN observations are collapsed with frequency weights. A pooled
mixture on a 257-point CP grid, with up to 2,048 additional narrow-mode proposals,
provides posterior-mean CPs. The optimizer has 500 iterations; finite-grid mean
weight score excess <=1e-8 certifies weights **on that grid**, not continuous CP
or the entire estimator. Budget-limited finite pilots remain usable and retain
that status. Compression, budgets, gaps and likelihood safeguards are serialized.

Sort by `(pilot_CP, chromosome_string, position_string)`; chromosome/position
strings are the normalized canonical values. String ties deliberately preserve
the prior estimator, rather than silently introducing numeric genomic ordering.
This order is frozen before all capacities and all subsamples. A cluster must be
one contiguous block, but the centers need not vary monotonically along the
chain. Distinct nonadjacent blocks can have identical centers.

## Native proposals

For K in 1..min(10,N), retain the K-1 largest-magnitude adjacent CCF differences;
exact ties prefer the earlier edge. This is Euclidean projection onto vectors
with at most K-1 nonzeros. The objective is negative marginalized likelihood plus
rho/2 times squared distance of adjacent differences to that set.

An MM step uses a boxed chain quadratic with posterior-mean complete-data
curvature and actual-objective line search. This positive curvature is **not**
the Hessian of the marginalized likelihood. CPU and CUDA maintain separate
implementations protected by opt-in differential tests. Host chain storage is
linear; active-set work can be quadratic in an adverse case.

Native bounds are [1e-8,1-1e-8] on CCF, 20 continuation levels, 300 iterations per
level, rho at most 1e12, and stationarity/constraint/block-range tolerances 1e-6.
These controls are shared with the exported native configuration. Raw diagnostics
are computed from the serialized CP round trip, because large rho amplifies tiny
CCF differences. A numerical-limit, finite-budget or stalled candidate can still
provide a useful partition. None certifies a constrained global optimum.

## Conditional refit and candidate search

Each block's CP is refitted independently on [0,p], including exact endpoints,
using 513 grid points, component modes and bounded scalar refinement of bracketed
maxima and endpoint intervals (`xatol=1e-12`, 500 iterations). This is a numerical
multimode search, not a global oracle. Merge only adjacent **exactly equal**
centers, preserving separated repeated visits.

Retain native partitions and eligible adjacent coarsenings. Fixed-center dynamic
programming optimizes ordered, nonempty block boundaries exactly in O(Nq),
including nonmonotone center visits and impossible (-infinity) assignments.
Alternate that step with conditional refits, preserving a better accepted
incumbent if a numerical refit decreases likelihood. Preserve the better
admissible score winner while adding proposals. Zero observed-mixture weights
require actual adjacent merges and refits, never a floor or silent relabeling.
Derived partitions keep raw-parent provenance without acquiring a raw status of
their own. Detailed candidate and polishing status fields remain in the tables.

## Score and output labels

For q occupied blocks with conditionally fitted centers c_k, fit nonnegative
simplex weights w_k to the **observed** mixture likelihood, separately from hard
chain memberships:

\[
\ell=\sum_i\log\left(\sum_k w_k L_i(c_k)\right),\qquad
S=-2\ell+(2q-1)\log N.
\]

Weights satisfy the existing 1e-8 mean fixed-center score checks. Published
occupied blocks have positive weights. This score is neither the conditional
block likelihood nor a penalty-augmented objective. It counts q CPs and q-1
weights. Conditional fitting does not jointly maximize this observed likelihood.
Every subsampled candidate is expanded to the full chain and scored on full N.

Choose the minimum score with existing deterministic tie rules (occupied q,
requested K, replicate for overall selection). Relabel final centers in descending
CP with stable ties. Label 0 means highest CP, not an imposed clonal component.
