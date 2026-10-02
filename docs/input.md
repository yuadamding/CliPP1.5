# Input contract: autosomal_inclusive_intervals_v1

Supply three whitespace-delimited text files. Headers are exact; extra columns
are inert except optional `mutation_id`. No file contains fitting truth.

| File | Required columns/content |
| --- | --- |
| SNV | `chromosome_index position ref_count alt_count` |
| CNA | `chromosome_index start_position end_position major_cn minor_cn total_cn` |
| Purity | Exactly one numeric token in (0,1], without a header |

Allowed chromosomes are integers 1–22, optionally prefixed by `chr` (case
insensitive). Leading zeros normalize away. X, Y, MT, 23, 24, fractional,
missing and other contigs fail explicitly. This is an **autosomal** model with
normal CN=2; do not recode sex chromosomes to autosomes.

Positions and CNA endpoints are positive integer int32 values. CNA intervals
are **1-based inclusive**, with start <= end. `[1,100]` and `[100,200]` overlap
and are rejected; `[1,100]` and `[101,200]` do not. All overlaps fail, including
same-CN duplicates or nested intervals. Segment input row order cannot select a
state. All files must share an assembly; `--assembly` records the user's label
and performs no lift-over. `unspecified` explicitly means no assembly was claimed.

Copy numbers must be finite integers: major >= minor >=0, major >=1, total =
major+minor, within int32. Missing CNA states are errors, not silently removed.
Each normalized SNV locus must be unique. Multiple alleles at one locus are not
supported by this legacy count format. Optional string `mutation_id` values must
be nonempty and unique; otherwise IDs are `chromosome:position`. IDs cannot
contain whitespace under this text schema. Literal missing markers (`NA`, `NaN`)
are not IDs.

Counts must be nonnegative integers, with positive total depth <= int32 max.
Malformed/missing/fractional counts and zero-depth rows are excluded with
`invalid_counts_or_zero_depth`; otherwise a locus outside CNA coverage is excluded
with `no_cna_segment`. Zero ALT with positive depth and high VAF are retained.
The first applicable reason is recorded, along with matched segment identity.
Excluding every row fails; the isolated failed attempt retains its ledger.
Structural errors (chromosomes, coordinates, ambiguous CN, purity, identities)
reject the whole input before fitting, rather than arbitrarily repairing it.

`input_ledger.tsv` reconciles each original 1-based SNV row, original encodings,
normalized coordinates, mutation ID, segment row, status and reason. `retained.tsv`
is the canonical fitted population, in original row order. `canonical_segments.tsv`
records normalized CN and original segment row IDs. Immutable copies and hashes
of all three supplied inputs are retained under `inputs/`.
