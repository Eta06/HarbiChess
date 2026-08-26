# OMURGA teacher qualification — 2026-08-26

## Decision

The search teacher gate **failed**. None of the 17 non-raw variants demonstrated
a positive independently verified action-value improvement with a positive 95%
bootstrap lower bound. The continuous learner and every new replay generation
remain blocked.

The completed result is stored at
`artifacts/qualifications/omurga-teacher-20260826-01/qualification.json` and is
published to the dashboard at `http://127.0.0.1:8765/`.

## Pre-registered qualification

- Baseline: `kilic-control-20260826-01` champion, model SHA-256
  `5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca`.
- Data: 32 deterministic positions selected from 2,969 validation records.
- Stratification: 19 represented combinations across opening/middlegame/endgame,
  low/medium/high branching, decisive/draw outcome, and repetition risk.
- PUCT budgets: 64, 128, 256, 512, and 800 simulations, each with noise off,
  noise on, and noise-target pruning.
- Gumbel Sequential Halving budgets: 64 and 128 simulations.
- Search repetitions: 2 fresh deterministic seeds per position and variant.
- Independent action verifier: clean 128-simulation continuation search.
- Uncertainty: 2,000 bootstrap samples for the verified action-value interval.
- Qualification gates: 95% action-value lower bound `> 0`, mean seed TV `<= 0.10`,
  and no value-MSE regression versus the raw network.

The command completed in 325.30 seconds:

```text
uv run harbichess-qualify-teacher \
  --run-result artifacts/runs/kilic-control-20260826-01/result.json \
  --shard artifacts/runs/kilic-control-20260826-01/replay/validation-00000.jsonl.gz \
  --output-dir artifacts/qualifications/omurga-teacher-20260826-01 \
  --positions 32 --workers 96 --seed 2026082619 \
  --search-budgets 64,128,256,512,800 --gumbel-budgets 64,128 \
  --search-repetitions 2 --verification-simulations 128 \
  --bootstrap-samples 2000 --maximum-stability-tv 0.10 \
  --maximum-value-mse-regression 0.0
```

## Results

| Variant | Verified action-value delta | 95% interval | Seed TV | Value MSE | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Raw network | 0.000000 | [0.000000, 0.000000] | 0.0000 | 0.475578 | baseline |
| Gumbel 64 | -0.000597 | [-0.001705, 0.000238] | 0.0005 | 0.475472 | fail |
| Gumbel 128 | -0.000627 | [-0.001809, 0.000192] | 0.0009 | 0.474983 | fail |
| PUCT 64 clean | -0.003765 | [-0.011223, 0.001057] | 0.0000 | 0.466618 | fail |
| PUCT 128 clean | -0.006409 | [-0.015757, -0.000918] | 0.0000 | 0.470166 | fail |
| PUCT 256 clean | -0.012374 | [-0.029895, -0.000920] | 0.0000 | 0.471599 | fail |
| PUCT 512 clean | -0.009256 | [-0.026427, 0.002843] | 0.0000 | 0.472434 | fail |
| PUCT 800 clean | -0.002111 | [-0.015234, 0.006532] | 0.0000 | 0.473591 | fail |

Noise-on PUCT was unstable at every budget: mean seed TV ranged from 0.1831 to
0.2489. Target pruning reduced that range to 0.1376–0.2089, but no pruned variant
met the 0.10 stability gate and all had negative mean verified deltas.

Gumbel was highly stable but changed the raw choice too little to establish a
positive teacher advantage (raw-argmax agreement 87.5% at 64 and 84.4% at 128).
Clean PUCT produced lower value MSE at several budgets, but lower root-value MSE
did not translate into stronger selected actions. At 128 and 256 simulations the
independent verifier found a statistically negative action-value delta.

This is the key diagnosis: with the current champion, position distribution, and
search implementation, increasing root simulations does not yet create a
qualified policy-improvement teacher. More learner training on these targets
would amplify an unproven or harmful policy signal.

## Inference and machine record

- Machine: MacBook Pro `Mac16,6`, Apple M4 Max, 14 CPU cores, 32 GPU cores,
  36 GB unified memory.
- OS: macOS 26.4 (25E241).
- MLX: 0.32.1.
- Qualification inference: 383,810 positions in 23,693 batches.
- Average/largest batch: 16.20 / 25 positions.
- MLX backend time: 99.65 seconds; mean queue wait: 16.11 ms.

## Max-ply target correction

Max-ply truncation is no longer serialized as a proven draw. Replay target schema
version 10 represents its value as unknown (`outcome_value = null`). Training keeps
the position's policy target but assigns zero weight to its value loss. Genuine
rule-terminal draws still use value 0. The same distinction is preserved when
arena continuation records reach their ply cap.

Legacy continuation/repetition target transforms are no longer part of the
default learning loop. They remain available only through explicit compatibility
flags so historical runs can be reproduced.

## Next decision

Do not implement the continuous learner yet. The next OMURGA step must diagnose
why clean search-selected actions fail independent verification—especially value
perspective/Q propagation, root action scoring, and verifier agreement—then repeat
this frozen qualification on a larger fresh stratified set. The gate must pass
before replay scale-up, learner/latest state, or a new self-learning generation.
