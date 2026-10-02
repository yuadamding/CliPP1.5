# Historical implementation only

`postprocess.R` retains historical fixed-multiplicity/lambda postprocessing for
attribution and audit. It rejects current `uniform_1_to_major_v1` intermediates.
The installed package never calls it. Reproduce old analyses with their pinned
revision and original inputs; do not apply its filters to current runs.

Original wrapper attributions: Kaixian Yu, Yujie Jiang, Shuangxi Ji, Yuxin Tang
(2021). The original CliPP introduction referenced
https://www.biorxiv.org/content/10.1101/2021.03.31.437383v2 . This historical
reference does not establish the numerical or scientific claims of the current
conditional fixed-chain estimator. Existing AGPLv3 licensing remains unchanged.
