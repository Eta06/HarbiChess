# KRITIK frozen joint policy/value transfer preregistration

Date: 2026-08-30

## Decision being tested

AKTAR qualified the Full Gumbel teacher and its target provenance, but every policy-only transfer
lost two baseline-solved forced defenses under search. The value head remained almost uniform on
held-out KOPRU replay (`WDL CE 1.09893`, expected-value standard deviation `0.00140`). KRITIK tests
whether corrected, balanced value supervision can make the critic useful while transferring the
qualified policy teacher. Search allocation, replay split, model size, and Full Gumbel settings stay
frozen.

This is not a continuous-learning or promotion run. Failure cannot be overridden by combined loss,
longer training, or a post-result threshold change.

## Frozen inputs

- baseline: `artifacts/runs/kopru-qualified-replay-20260828-01/baseline/model.safetensors`;
- baseline SHA-256: `5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca`;
- policy targets: `artifacts/diagnostics/aktar-full-gumbel-targets-20260828-05/result.json`;
- replay: KOPRU schema-12 train and validation shards;
- policy rows: 384 train and 192 validation;
- value rows: every known terminal-outcome row in its original game-disjoint split;
- max-ply rows with `outcome_value=None` receive exactly zero value weight and can never enter the
  value sampler;
- continuation ranking set: 32 fixed stratified validation positions, seed `2026083017`;
- no continuation/repetition target transform.

Before optimization, provenance must prove that schema 12 is in use, all unknown max-ply targets are
excluded from value supervision, decisive game labels alternate with side to move, draw labels stay
zero, train/validation game identities are disjoint, and the qualified policy-target identities and
model hash match their recorded sources.

## Frozen optimization

### Stage A: value-head learnability control

- start from the baseline;
- freeze trunk and policy head;
- train only `value_conv`, `value_hidden`, and `value_output`;
- sample known outcomes equally across win/draw/loss, then game-balance within each class;
- Adam, learning rate `5e-4`, batch 64, maximum 400 steps;
- validate every 20 steps and stop after six non-improving validations;
- select strictly by macro validation WDL cross-entropy.

Stage B is blocked unless Stage A improves macro WDL CE by at least `0.10`, produces ordered mean
expected values `win > draw > loss` with both adjacent margins at least `0.03`, reaches expected-score
Pearson at least `0.20`, improves multiclass Brier by at least `0.03`, and has no non-value parameter
change.

### Stage B: joint transfer

- restart from the Stage-A selected checkpoint;
- unfreeze the whole network;
- every step uses one outcome/game-balanced value batch of 64 and one game-balanced qualified-policy
  batch of 64;
- optimize `policy CE + value CE` with equal weights using AdamW, learning rate `1e-4`, weight decay
  `1e-4`, and gradient clipping at 5;
- maximum 400 steps, validation every 20 steps, patience six;
- checkpoint selection is lexicographic: it must first satisfy all value and policy validation
  constraints, then minimize macro WDL CE. Combined loss alone cannot select a checkpoint.

## Frozen pre-arena gates

The selected joint checkpoint must satisfy all of the following:

1. validation micro and macro WDL CE each improve by at least `0.10` from baseline;
2. multiclass Brier improves by at least `0.03`, ECE-10 is at most `0.12`, expected-score Pearson is
   at least `0.20`, and the win/draw/loss means remain correctly ordered with `0.03` adjacent margins;
3. train-to-validation macro WDL CE gap is at most `0.15`;
4. Full Gumbel policy validation CE improves by at least `0.05`, top-action agreement improves by at
   least two percentage points, and validation agreement trails train by at most 15 points;
5. on the frozen continuation set, independent depth-4 tactical/material action-value ranking has
   mean Spearman improvement at least `+0.05`, candidate mean Spearman is positive, and verified
   top-action agreement does not regress;
6. raw tactical solve count does not regress; 256-simulation Full Gumbel solves at least 4/8 and
   retains every baseline-solved case;
7. losses and gradients remain finite.

Only a checkpoint passing all seven gates may enter one fresh, preregistered paired search arena:
32 opening pairs, eight opening plies, 128 Full Gumbel simulations, maximum 256 plies, seed
`2026083029`. Continuous learning requires a positive paired expected-score 95% lower bound, no
avoidable-threefold regression, no decisive-score regression, and a healthy baseline-vs-baseline
control. Promotion and new generation remain unauthorized in this experiment.

If Stage A or Stage B fails, KRITIK stops. The next action is a structural audit of value-target
distribution, replay balance, representation/head capacity, and loss interference—not a longer run.
