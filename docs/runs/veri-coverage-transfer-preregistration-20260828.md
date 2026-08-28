# VERI teacher-coverage transfer preregistration (2026-08-28)

## Hypothesis

AKIS learned its 96 training targets but failed to generalize. Increasing the
number of fresh, stratified teacher-labelled positions while holding learner
compute fixed should improve out-of-sample transfer if label coverage is the
actual bottleneck.

## Frozen design

- Source: existing KOPRU qualified replay; no new self-play generation
- Exclusions: every TERAZI, DOKU, OLCEK, and BAG diagnostic identity
- Fresh split: 384 training and 96 validation positions
- Search: clean 512/800 simulations; existing uncertainty and verifier gates
- Dataset seed: `2026082834`; bootstrap seed: `2026082835`
- Learner: AKIS rank-8 mergeable policy adapter
- Learner seed: `2026082836`
- Learner compute: unchanged 480 steps, batch 16, AdamW `2e-4`
- Checkpoints: unchanged 0, 60, 120, 240, 480

All AKIS transfer gates remain unchanged: 5% validation cross-entropy
improvement, teacher Spearman at least 0.35, positive paired verified-gain lower
bound, harmful ratio at most 10%, regret at most 0.10, best-action top-16
coverage at least 80%, bitwise-identical WDL, and no tactical solve regression.

Failure does not authorize more training, a threshold change, arena,
generation, or promotion. Passing authorizes only a separate frozen search
qualification of the selected checkpoint.
