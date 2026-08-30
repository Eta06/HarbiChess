# DEVRIYE pooled value reservoir preregistration

## Prior result

One-step value batches of 64, 256, 1024, and 2048 on a single generation all
failed the joint old/fresh gate. Larger batches consistently improved fresh
held-out CE but left small negative Pearson drift and did not satisfy every old
micro/macro/Pearson constraint. Larger same-generation batches are rejected.

## Hypothesis

A continuous learner should not force a value update from every small correlated
generation. Policy can advance while the qualified MIHVER value remains frozen;
value should update only when a reservoir contains enough independent terminal
games to produce a Pareto-improving gradient.

## Frozen cached audit

- Pool all known-terminal replay from the three accepted `-13` updates (88
  independent games) without reusing max-ply rows.
- Deterministically split games 75/25 within outcome into fit and held-out sets.
- Start from immutable MIHVER and run exactly one value-only Adam update at
  learning rate `1e-4`.
- Test total batches 1024, 2048, and 4096, each half historical and half pooled
  fresh. Fresh sampling remains 25% loss / 50% draw / 25% win and game-balanced.
- No policy update, extra value step, anchor, new replay, or metric threshold
  change is allowed.

## Gate

Use the unchanged joint requirement: old fixed-validation micro CE, macro CE,
and Pearson cannot regress; pooled fresh held-out CE must improve without fresh
Pearson regression; outcome margins and finite gradients must remain healthy.

Choose the smallest passing batch. If none passes, reject reservoir scale alone
and move to value target/objective or representation changes. Cached evidence
cannot authorize production generation.
