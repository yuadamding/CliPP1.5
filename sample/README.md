# Legacy-format smoke sample

The three original `sample.*.txt` files are retained unchanged from commit
15d397b680ed3392a2ec3bbead1e5d5b460448ff. Their earlier biological provenance and
true memberships/multiplicities are not established here; **do not invent truth**.
Purity is exactly 0.9. Coordinates are treated as autosomal, 1-based and inclusive
under the new input contract; genome assembly is unspecified.

`tests/fixtures/legacy_cpu_baseline.json` records input hashes, expected retained
chain order/pilot and all ten capacities' labels, CPs, mixture weights and scores,
obtained by running the frozen original source before refactoring. The example
retains all 487 SNVs. The input matching map, IDs and retained/excluded ledger are
published by each verified run. `tests/fixtures/synthetic/` is a separate,
explicitly generated fixture with truth and a generator receipt.
