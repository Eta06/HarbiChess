# UFUK: experimental one-ply targets and prior-anchored objective

Status: **39 targeted tests passed; no production online training or strength
claim.** Current four-run capacity experiment continues from its frozen f4c10e5
source, controller, data and gates. These new modules do not modify that source
or an existing MLX/Torch learner, replay schema, checkpoint or default.

Motivation: after failed broad Stockfish/depth contrasts, prepare a genuine
fresh-self-play alternative to repeated imitation. The primary prior-directed
paper and pinned released-code audit are in
[research](../research/UFUK-prior-directed-code-audit-20261004.md).
The prior/EMA mechanism is established research, not HarbiChess novelty. Its
reported strong starting network is substantially different from ours; this
implementation has not established a gain with our weak prior.

## Exact semantics

`one-ply-stm-wdl-v1` preserves the root FEN and full legal history, action,
explicit draw-claim convention, rollout cutoff, observed terminal result and
termination, target source, recorded policy/proposal probabilities, clipping cap,
importance, mover WDL and expected-score advantage. Real terminal targets use
the mover's outcome. Nonterminal targets swap the child EMA W/L and retain D,
including when rollout is capped. Bootstrap labels are not observed outcomes.
Terminal precedence over a simultaneous cutoff is explicit. Actions after a
terminal/claimable position are rejected under the selected convention.

`prior-anchored-one-ply-v1` uses:

```
rho = min(explicit_cap, recorded_pi(a|s) / recorded_mu(a|s))
A = E(target_mover_WDL) - E(recorded_online_pre_WDL)
E = W + D/2 = (Q + 1)/2; Q = W-L
loss = mean(-rho*A*log pi(a|s))
     + beta*mean(KL(base_legal_policy || pi))
     + c_v*mean(rho*KL(target_mover_WDL || online_WDL))
     + c_base*mean(KL(base_WDL || online_WDL))
```

Coefficients must be explicitly supplied. Test coefficient values are not a
registered training choice. Support and probability normalization are checked;
illegal logits are masked, including values larger than every legal logit.
Zero-probability KL terms are exactly zero rather than `0*log(0)`/`0*(-inf)`.
Copied read-only target arrays have no external aliases or gradient graph.
Unequal microbatches must be weighted by row count, followed by one optimizer
update; no equal-chunk averaging or replay epochs are implied by this objective.

## Actual verification

Command from `/workspace/HarbiChess`, source bytes embedded in the companion
evidence file:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q \
  tests/test_online_targets.py tests/test_online_objective.py
```

Result: **39 passed in 0.96s, zero skips**, reported pytest time rather than a
whole-process performance benchmark. Ruff check/format passed for all six new
source/test files. Tests executed actual Torch autograd and MLX CPU autodiff,
checked all five loss components against an independent NumPy float64
log-sum-exp oracle, Torch float64 gradients against finite differences, and
Torch/MLX float32 gradients with absolute tolerance2e-6. Also checked actor
gradient direction, illegal zero gradients, zero-importance behavior, unequal
microbatch loss/gradient accumulation, target ownership and invalid inputs.

Rule probes cover white/black mating actions, a material draw, ghost-free
nonterminal history, announced repetition and fifty-move claims, optional
unclaimed convention, cutoff versus terminal targets, and subnormal proposal
probability clipping. Scalar importance cannot by itself certify that two full
distributions have the same support; the future actor must record and validate
their complete sampling provenance.

Existing full suite587PASS/54.47s/zero skips predates these new modules. It is not
presented as a post-change full-suite result; the broader regression suite will
run when no registered training/speed measurement is active. Targeted log:
`work/ufuk-online-target-objective-tests-20261004.log`.

## Limits and next decision

No optimizer update, rollout, replay, policy improvement, calibration improvement,
paper reproduction, teacher superiority or strength result occurred here.
Apple Metal/CUDA/BF16 remain device-untested. CPU loss/gradient parity does not
establish cross-framework AdamW or complete online resume equivalence.

An actual loop still needs owned current/base/EMA models, Adam state, immutable
book/source/config hashes, every active game's complete history and cursor,
all used RNG states, counters, atomic checkpoints and a fresh-process next-rollout
plus next-update equality check. Old replay/checkpoint formats are preserved;
there is no migration or weights-only import masquerading as full resume.
Any production learning requires a new prospective budget, fixed controls/seeds,
initial/final independent strength games, retention and stopping gates.
