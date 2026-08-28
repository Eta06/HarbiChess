# ESAS system teacher qualification preregistration (2026-08-28)

## Hypothesis

The release network's clean PUCT search is a useful policy-improvement operator
even though its complete legal-action Q ranking does not correlate strongly with
the handcrafted depth-4 verifier. The correct test is whether search improves
the same frozen network's raw policy in play and in rule-verifiable tactics.

This experiment does not create replay, train a learner, or authorize a release.

## Frozen inputs

- Model: SHA-256
  `5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca`
- Path: `artifacts/runs/kopru-qualified-replay-20260828-01/baseline/model.safetensors`
- Network: 16 trunk channels, 2 residual blocks, 4 policy channels,
  2 value channels, 32 value hidden units
- Search: clean PUCT, `c_puct=1.5`, no Dirichlet noise, deterministic move
  selection, draw claims enabled
- Budgets: raw policy and 64/128/256 clean simulations
- Seed: `2026082867`
- Openings: 32 unique, deterministically sampled 8-ply opening prefixes
- Games: 64 color-balanced games per search budget against raw policy
- Maximum length: 256 plies; max-ply outcomes are reported separately and do
  not become value targets
- Workers: 24; shared MLX batch evaluator, maximum batch 24, wait 0.25 ms
- Tactical suite: the existing eight independently rule-verified mate,
  defense, and hanging-piece cases at raw/64/128/256

Raw policy means the legal policy argmax from one direct network evaluation. It
must not be implemented as a noisy search or as an oracle evaluation.

## Frozen measurements

For each budget record wins/draws/losses, expected score, paired opening score
deltas, bootstrap 95% interval, decisive score, max-ply rate, threefold rate,
games/hour, positions/second, MLX batch statistics, tactical solve count, and
per-case regressions. Full-action verifier rank correlation remains descriptive
only and is not recomputed for this qualification.

## Gate

The system teacher qualifies only if all conditions hold:

1. 128- and 256-simulation expected score versus raw policy are both above
   0.55, and the paired bootstrap 95% lower bound for 256 is above 0.50.
2. The 256-simulation paired mean score is no worse than 128 by more than 0.02.
3. The 256-simulation tactical solve count is at least two cases above raw and
   is not below the 128-simulation count.
4. No rule-verifiable tactical case solved by 128 becomes unsolved at 256.
5. The 256-simulation decisive score is at least 0.50; strength may not be
   manufactured solely by converting losses to draws.
6. Its max-ply rate is at most raw plus 0.10 and its threefold rate is at most
   raw plus 0.10.
7. Every game is paired by identical opening and opposite colors, all model and
   configuration hashes match, and all recorded values are finite.

The 64-simulation arm measures budget scaling but is not a strength gate. Sample
size and thresholds may not change after results are visible.

## Decision after the run

- Pass: authorize implementation of persisted learner/latest and rolling replay,
  still with no large generation or release promotion.
- Fail with tactical or strength regression: keep learner closed and isolate
  value/search allocation using the failed cases.
- Fail only through repetition/max-ply: fix game-horizon/search behavior before
  learner work; do not reintroduce continuation target heuristics.
