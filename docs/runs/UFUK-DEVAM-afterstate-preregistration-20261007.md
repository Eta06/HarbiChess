# UFUK-DEVAM: selected-action afterstate target ablation

Registered before actual conversion, proof, fit, or matches for this variant.
The original ownQ-v2 known160 experiment failed both unchanged strength screens.
Independent source review found no verified root-Q viewpoint/sign bug. Regressing
the root search backup is a valid distillation target; changing the target board
is a prospective learning hypothesis, not a correction of a demonstrated bug.

Hypothesis: training the legally played selected move's child position against
the root search backup can give the evaluator credit at a position it evaluates
when selecting actions. It may also harm strength: the root backup has a shorter
effective horizon than a fresh search from the child. This is established
afterstate/value learning, not an originality claim.

Use exactly the two fixed seeds 20262905 and 20262906 and the same sealed 1,024
chronological eligible TRAIN rows per seed from ownQ-v2. Replay all original
events, censored episodes, protected exclusions, full legal histories and ordered
search-alias sidecars through the original converter. No additional teacher
queries or reanalysis searches generate labels. Unplayed protected/discarded
actions remain in source replay and never become training rows. For each eligible
actually played action, encode the resulting child position. Nonterminal target
is `-clip(root mover search Q, -1, 1)`; terminal target is exact child-side WDL.
UNKNOWN source endings remain disclosed, rather than being relabeled as draws.

Initialize from the exact same-seed frozen teacher-parent MODEL weights, with
fresh Adam and global/private RNG state. This remains a disclosed teacher
bootstrap; the new labels come from own search. This is not a full resume of
teacher training. Retain original old-phase checkpoints and raw replay unchanged.
Use a distinct afterstate dataset, native schema and training phase. Fixed 64
updates, batch 256, Adam lr .001, betas .9/.999, eps 1e-8, weight decay 0,
gradient norm cap 5; no loss-based stopping or checkpoint selection. New immutable
ROOT clocks: conversion <=600 seconds; proof <=600; fresh64 <=1,800. First prove
whole8 equals pause4 plus fresh-process resume8 in the complete native payload,
including optimizer and RNG, with six actual strict loads; then require fresh
strict loads at fit0 and fit64. These checks do not establish strength.

Each exact new child must pass the same 48 TRAIN C/Torch and role-rotated search
latency profile, then all ten fixed known-eight paired-color arms (160 games)
against E8, its exact frozen parent and Stockfish19, with the unchanged original
512-node/q2/max8 search, SF Threads1/Hash16/ClearHash and actual UCI node overruns
logged. Full histories and six actual search packets/traces must be audited.
Known development results are only a screen. No thresholds, caps or statistical
requirements change after observing outcomes. Both seeds must be eligible before
the single authorized new independent 960-game confirmation cohort can be drawn.
The old bootstrap bounds and prospectively registered exact-betting bounds both
remain required. No formal book has been drawn for a failed candidate.

The frozen afterstate source and separate typed arena wiring are under
`experiments/ufuk/continuation-20261007/afterstate-own-v1` and
`variant-known160-v2-afterstate`. Copied historical protocols in the latter are
explicit parent examples, not usable afterstate registrations. ROOT must create
new exact child protocols and observed clocks. This phase uses existing CPU
allocation only; no GPU, SSH or paid resources. Success requires independently
verified repeatable strength gain, never conversion/proof/loss alone.
