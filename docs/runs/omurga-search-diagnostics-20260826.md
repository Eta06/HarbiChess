# OMURGA search diagnostics — 2026-08-26

## Decision

No implementation defect was found in value perspective, terminal scoring, Q
backup, root PUCT convention, history reconstruction, or verifier sign handling.
The current search is not a qualified teacher because the frozen champion's value
head is almost constant and its policy is diffuse. With little leaf separation,
PUCT spends the available budget opening almost every legal child and does not
produce reliably better root actions.

The continuous learner and every new replay/training generation remain blocked.
No MCTS behavior, target, replay, learner, or model weight was changed in this
stage.

Final artifact:
`artifacts/diagnostics/omurga-search-20260826-03/diagnostics.json`.

## Frozen setup

- Champion SHA-256:
  `5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca`.
- Network: 16 trunk channels, 2 residual blocks, 4 policy channels, 2 value
  channels, and a 32-unit value hidden layer.
- Replay sample: 32 deterministic stratified positions from
  `kilic-control-20260826-01` validation replay.
- Tactical suite: eight rules-proven positions covering two mate-in-one, two
  mate-in-two, two forced-defense, and two hanging-piece cases.
- Clean budgets: 1, 4, 8, 16, 32, 64, 128, 256, 512, and 800 simulations.
- Root noise: disabled. Training, replay generation, and learner: disabled.
- Machine: Apple M4 Max, 32-core GPU, 36 GB unified memory; MLX 0.32.1.

The final run completed in 65.80 seconds and evaluated 61,147 positions in 6,643
MLX batches.

## Convention audit

| Area | Result | Evidence |
| --- | --- | --- |
| Value perspective | pass | Neural WDL is interpreted as side-to-move win minus loss. |
| Terminal value | pass | A checkmated side-to-move evaluates to `-1`. |
| Q backup | pass | Leaf `-1` backs up root-to-leaf as `+1,-1,+1,-1`. |
| Parent action value | pass | Root move statistics expose `-child.mean_value`. |
| PUCT scoring | pass | Exploitation is parent-perspective Q; exploration uses prior and visit count. |
| FPU/unvisited child | observed | Unvisited child Q is exactly zero. This is intentional, not a sign bug. |
| History/repetition | pass | Full move history claims threefold; identical FEN without history does not. |
| Transpositions | isolated | No transposition table exists, so incompatible histories cannot merge. |
| Virtual loss | isolated | Trees are not shared; batching only coalesces independent leaf inference. |
| Verifier convention | pass | A mating child returns `+1` to the parent action verifier. |

The FPU value of zero is particularly important here. It is a valid convention,
but with a near-zero value head it makes unvisited actions look almost identical
to visited actions. Changing FPU could alter allocation, but it would be a new
search experiment rather than a correction to an implementation error.

## Batching audit

Serial single-position and coalesced MLX batches are not bitwise identical under
BF16. The maximum observed differences were `0.00004054` in a legal policy
probability and `0.00004028` in value. Both are within `1e-4`.

More importantly, serial and batched 64-simulation searches selected the same move
on all eight tactical positions. There was no virtual-loss interaction or shared
tree. The measured numerical variation is therefore not the primary teacher
failure in this run.

## Tactical budget sweep

Every expected move is independently proved from chess rules: checkmate at the
specified horizon, the only defense avoiding mate-in-one, or the unique maximum
immediate material capture.

| Simulations | Solved | Regressed cases |
| ---: | ---: | --- |
| Raw policy | 1 / 8 | — |
| 1 | 1 / 8 | — |
| 4 | 0 / 8 | forced-defense-a |
| 8 | 3 / 8 | — |
| 16 | 5 / 8 | — |
| 32 | 4 / 8 | mate-in-two-a |
| 64 | 5 / 8 | — |
| 128 | 7 / 8 | — |
| 256 | 7 / 8 | hanging-queen |
| 512 | 7 / 8 | — |
| 800 | 7 / 8 | — |

This rules out a globally reversed value or broken terminal propagation: search
finds mates and forced defenses and improves aggregate solve rate substantially.
It is not strictly monotonic, however. The hanging queen is found at 64/128 and
lost again from 256 onward, while the hanging rook becomes correct. That is the
signature of unstable leaf ranking rather than a terminal-value failure.

## Value calibration finding

The value head is effectively collapsed on the 32 replay positions:

- all 32 predictions are negative;
- minimum/maximum: `-0.02516` / `-0.01854`;
- mean: `-0.02113`;
- standard deviation: `0.00150`;
- outcome correlation: `0.0542`;
- value MSE: `0.47569`.

Mean prediction by eventual side-to-move outcome:

| Outcome | Mean predicted value |
| --- | ---: |
| Win | -0.02100 |
| Draw | -0.02118 |
| Loss | -0.02121 |

The head therefore cannot distinguish a win, draw, or loss in this sample. Its
small negative constant is close to an uninformative zero predictor. Increasing
MCTS depth cannot manufacture a reliable Q ordering when non-terminal leaves all
receive nearly the same value.

## Root allocation finding

The raw policy is also broad: mean top-action prior is only `7.78%`. At 64
simulations, search visits an average of `25.38` out of `25.69` legal children.
The leader receives only `6.06` visits (`9.47%` of budget) and leads the runner by
less than one visit on average. It is the highest-Q action in only `9.38%` of
positions.

| Simulations | Leader visits | Budget share | Visit margin | Leader is best Q | Raw agreement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 6.06 | 9.47% | 0.91 | 9.38% | 12.50% |
| 128 | 13.06 | 10.21% | 3.22 | 21.88% | 15.63% |
| 256 | 28.31 | 11.06% | 8.41 | 56.25% | 25.00% |
| 512 | 66.44 | 12.98% | 26.81 | 75.00% | 25.00% |
| 800 | 112.59 | 14.07% | 48.59 | 68.75% | 25.00% |

Higher budgets eventually concentrate visits, but the value ordering remains
weak enough that the best-Q ratio falls again at 800. This matches the previous
teacher qualification: more simulations change policy, but the independently
verified action-value interval does not become positive.

## Root cause and next safe diagnostic

The failure is not primarily a search sign or rules bug. It is the combination of:

1. an almost non-discriminating value head;
2. broad policy priors;
3. zero FPU causing nearly all children to consume early simulations when Q is
   flat;
4. a 64–800 simulation budget that is still shallow after division across roughly
   26 legal actions;
5. a verifier using the same weak leaf evaluator, which is convention-compatible
   but cannot serve as an independent strength oracle.

The next step should remain diagnostic: hold the champion and position set fixed,
replace only the leaf value in an isolated counterfactual with a deterministic
shallow tactical/material oracle, and test whether PUCT action improvement and
monotonicity return. That would distinguish search allocation limitations from
the collapsed learned value signal. It must not write replay, train weights, or
authorize a generation. FPU or PUCT tuning should only follow if that controlled
counterfactual shows the leaf evaluator is no longer the dominant bottleneck.
