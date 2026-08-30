# DEVRIYE value stability-plasticity ablation preregistration

## Problem

The fresh `-13` chain accepted all three relative update gates and scored 0.53125
against MIHVER, but tiny permitted per-update drift accumulated past exact final
WDL and continuation no-regression gates. Production integration remains blocked.

## Frozen cached ablation

- Reuse only `-13` update-1 replay for diagnosis; it cannot authorize generation.
- Split known-terminal games deterministically by game identity: 75% fit and 25%
  game-disjoint validation, stratified by outcome where possible.
- Keep the existing 32 historical / 32 fresh value batch and one local update.
- Add KL distillation from immutable MIHVER WDL logits on the historical half.
- Test anchor weights `0.25`, `1`, `4`, and `16`; include unanchored `0` as the
  control. No weight may be added after results.
- Report old fixed validation micro/macro CE, Pearson and ECE, plus fresh held-out
  CE, macro CE, Pearson and per-outcome separation.

## Gate

An arm is useful only if it simultaneously:

1. does not regress old-validation micro CE, macro CE, or Pearson versus MIHVER;
2. improves fresh held-out micro CE without reducing fresh Pearson;
3. retains both MIHVER outcome margins and finite gradients;
4. uses no evaluation row as a training or distillation row.

If no arm passes, reject output anchoring and audit the value target/update
objective rather than weakening final chain gates. If an arm passes, separately
preregister a fresh chain with that fixed weight and cumulative baseline gates.
