# Limitations and interpretation

- Single sample, autosomal input, diploid normal CN, fixed purity and integer
  major/minor CN. No mixed-CN-state model, sex-chromosome normal-CN model,
  multi-region phylogeny or purity inference is supplied.
- The fixed chain can make biologically correct memberships inexpressible. The
  capacity cap is ten, and rare/close clusters can be missed. A better BIC-form
  score need not imply better mutation membership or CCF error.
- Finite grids, numerical scalar refits, penalty continuation and candidate
  exploration provide scoped checks, not a global solution of the entire problem.
- Full multiplicity support is preserved. Python likelihood arrays can scale as
  N times maximum major CN. Cache limits do not bound all memory. The chain QP
  uses linear storage but can require quadratic active-set work.
- Zero fitted mixture weights require adjacent refitting; zero-weight occupied
  outputs are not allowed. Mixture weights, assignment proportions and cell
  abundances are different quantities.
- Conditional multiplicity probabilities assume selected CP, fixed purity/CN and
  partition. Subsample variability is not automatically a confidence interval.
- CUDA/compiler/device failures are not numerical success. CPU tests and CUDA
  skips cannot qualify GPU execution. The Docker recipe is unexecuted until a
  receipt records both a real image run and independent output verification.
- Scientific performance needs protected cases, failures and denominators. The
  legacy smoke example has unknown truth. The included small simulation panel
  is not evidence of superiority on CN-first4K, SimClone or PhylogicNDT500.

When a fit looks poor, distinguish candidate-search failure (a better feasible
partition was missed), scoring failure (the score preferred a scientifically
worse solution), and representation failure (the frozen chain cannot express the
truth). Exhaustive small cut enumeration is a useful reference, but conditional
center refitting remains numerical; do not call it a global oracle.
