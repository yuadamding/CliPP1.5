# Final source path comparison

Production fingerprint: `a3aa1dc7719b743e0e9342291aa25dd21f7860632ade008a67768d8d5c40cf30`.

Single observed fits are not repeated latency measurements. Incomplete baseline searches remain incomplete.
Saved-QP replay and full-path qualification are separate claims; neither substitutes for the other.

| Fixture | Starts | Penalties complete | Final publication | Baseline selected estimates | Fit seconds, current / baseline |
| --- | --- | --- | --- | --- | --- |
| below_one N256 | 98/99 | 25/26 | False | no baseline | 56.760 / unavailable |
| mixed_support N256 | 99/99 | 26/26 | True | exact | 66.694 / 169.782 |
| mixed_support N512 | 99/99 | 26/26 | True | no baseline | 58.302 / unavailable |
| below_one N64 | 99/99 | 26/26 | True | exact | 29.691 / 57.850 |
| mixed_support N64 | 99/99 | 26/26 | True | exact | 100.223 / 95.370 |

| Existing qualifier case | Current seconds | Baseline seconds | Ratio | Selected estimates |
| --- | ---: | ---: | ---: | --- |
| scaling N16 | 4.619 | 4.481 | 1.031 | differs |
| scaling N64 | 7.464 | 9.975 | 0.748 | exact |
| scaling N256 | 83.333 | 76.388 | 1.091 | exact |
| single_support eager | 6.519 | 6.181 | 1.055 | exact |
| single_support compiled | 2.284 | 2.073 | 1.102 | exact |
| mixed_multiplicity eager | 42.752 | 40.613 | 1.053 | exact |
| mixed_multiplicity compiled | 32.236 | 30.367 | 1.062 | exact |
| all_bounds_below_one eager | 205.738 | 197.100 | 1.044 | differs |
| all_bounds_below_one compiled | 187.580 | 178.051 | 1.054 | exact |

| Saved-QP replay | Receipt | Resolved | Unresolved | Full-path authority |
| --- | --- | ---: | ---: | --- |
| replay-later-below64 | passed | 1 | 0 | none |
| replay-original-below64 | passed | 2 | 0 | none |
| replay-first-below256 | failed | 0 | 1 | none |
| replay-largest-mixed256 | passed | 1 | 0 | none |
| replay-original-mixed256 | passed | 11 | 0 | none |
