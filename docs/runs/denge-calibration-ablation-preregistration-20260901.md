# DENGE calibration ablation preregistration

Date: 2026-09-01

## Status and scope

PUSULA-16 remains a failed production-readiness experiment. Its only failed
preregistered cumulative check was `fresh_ece_noninferior`; this document does
not reinterpret that run or change any PUSULA margin.

DENGE tests one mechanism suggested by the frozen PUSULA-16 logits: the rolling
candidate is underconfident. On the full held-out fresh set, lowering the scalar
temperature from 1.0 to 0.5 left class decisions unchanged while reducing ECE-10
from 0.123133 to 0.011066, WDL CE from 0.692119 to 0.642307, and Brier from
0.415157 to 0.387365. This observation may select the mechanism, but it may not
qualify production.

## Frozen diagnostic ablation

- Network: the immutable PUSULA-16 update-003 checkpoint.
- Calibration family: one positive scalar shared by all three WDL logits. No
  policy, representation, class bias, replay weight, exposure, search, or
  learner parameter may change.
- Initialization: exact identity scale, so wrapping MIHVER remains bitwise
  function preserving.
- Fit/test partition: deterministic and game-disjoint. The scalar is selected
  only by minimum micro WDL CE on the fit partition. Test labels are not used in
  scalar selection.
- Search and verifier configuration: identical to PUSULA-16.
- PUSULA-16 final data is diagnostic-only because its aggregate result has
  already been inspected. A passing DENGE diagnostic authorizes integration and
  a wholly fresh cumulative pilot; it cannot promote or qualify the checkpoint.

The diagnostic passes only if all checks hold:

1. Test ECE-10 improves by at least 0.020.
2. Test CE and Brier do not regress; expected-score Pearson does not regress by
   more than 0.005.
3. On the old diagnostic distribution, CE and Brier deterioration is at most
   0.003, Pearson deterioration is at most 0.010, and absolute ECE-10 is at most
   0.120.
4. Policy logits are bitwise unchanged.
5. Full Gumbel loses no previously solved tactical case and still solves at
   least 5/8 cases.
6. Continuation mean Spearman deterioration is at most 0.020 and verified-top
   agreement loses at most one of 1,440 positions.

Failure of any check rejects this calibration mechanism as implemented. The
thresholds will not be changed after observing the ablation.

## Fresh cumulative proof after integration

If the diagnostic passes, temperature fitting will use a game-disjoint
calibration partition excluded from gradient fitting and tuning selection. A new
continuous pilot must use fresh replay, fresh qualification starts, and a fresh
seed. It retains the PUSULA preregistered cumulative margins and power floors:

- old qualification: 1,536 attempts, at least 384 known games;
- fresh qualification: 2,688 attempts, at least 744 known games;
- 20,000 game-paired bootstrap resamples at 95% confidence;
- old CE/Brier margins 0.003, macro CE 0.005, Pearson 0.010, ECE 0.010,
  absolute ECE at most 0.120;
- fresh CE improvement at least 0.002, non-negative macro CE/Brier/Pearson
  improvement, ECE deterioration at most 0.020, absolute ECE at most 0.150;
- the existing final arena, tactical, continuation, integrity, and exact-resume
  gates remain unchanged.

No production continuous loop, generation, or promotion is authorized until
that wholly fresh experiment passes every gate.
