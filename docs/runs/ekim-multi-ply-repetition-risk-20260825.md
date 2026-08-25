# EKIM multi-ply repetition-risk ablation — 2026-08-25

## Decision

The v5 candidate did not satisfy the frozen EYLUL promotion gate. No new generation
was started and the champion chain is unchanged. The candidate preserved v4's point
strength and improved its avoidable-threefold point estimate, but neither the paired
strength interval nor the avoidable-threefold uncertainty cleared the required gate.

## Target change

Target schema v5 extends, rather than rewrites, the immutable v4 branch-value evidence.
Schema versions 3 and 4 remain readable. Each previously qualified non-repeat branch
now stores:

- a three-ply horizon;
- rollout and repetition-event counts;
- the observed repetition-risk rate;
- a one-sided 95% Wilson upper confidence bound.

The audit used 16 champion-guided stochastic rollouts per branch and 32 MCTS
simulations per rollout step. A repetition event means that, at rollout depth two or
three after the selected non-repeat branch, the current position is a twofold return or
a threefold claim is available. Search did not claim optional draws during the probe,
so a one-ply claim did not prematurely hide a later two-to-three-ply return.

The maximum risk upper bound was frozen at 20% before the audit. With 16 rollouts,
zero events has a 14.46% upper bound and passes; one event has a 23.75% upper bound and
fails. Qualified policy mass retains the v4 lower-confidence value surplus and is
additionally weighted by one minus the branch risk upper bound.

## Replay audit

Source: the immutable 55-root AGUSTOS v4 shard, generated from the champion at
`artifacts/runs/haziran-candidate-20260825-01/baseline/model.safetensors`.

| Metric | Result |
| --- | ---: |
| Source roots | 55 |
| Source qualified branches | 62 |
| Branches with 0 / 16 events | 56 |
| Branches with 1 / 16 events | 5 |
| Branches with 2 / 16 events | 1 |
| Accepted v5 roots / branches | 49 / 56 |
| Audit time | 52.30 s |
| Neural positions | 91,405 |
| Average MLX batch | 29.07 |
| MLX backend time | 17.53 s |

Artifact: `artifacts/audits/ekim-repetition-risk-20260825-01/`.

## Fixed-compute training

The ablation used the same HAZIRAN champion, fresh train/validation shards, random
seed `2026082504`, 200 attempted steps, batch size 64, and 25% continuation sampling
used for v4. Training duration and continuation exposure were not increased.

| Metric | v4 | v5 |
| --- | ---: | ---: |
| Continuation records | 55 | 49 |
| Attempted steps | 200 | 200 |
| Restored checkpoint | 180 | 190 |
| Best validation loss | 7.24649 | 7.26792 |
| Final train loss | 5.75100 | 5.77321 |
| Training time | 1.81 s | 2.01 s |

The v5 run stopped because the fixed 200-step compute limit was reached, not because
of early stopping. It restored the best-validation checkpoint at step 190 before the
arena. Validation loss was recorded but was not used as the promotion decision.

## Frozen paired arena

The candidate played the exact same 200 color-balanced opening assignments as the
continuation-off and v4 arms: 32 games at seed `2026082552`, 64 at `2026082562`, and
104 at `2026082572`. Every game used 32 simulations, 12 opening plies, and a 256-ply
limit.

| Arm | W-D-L | Score | Avoidable threefold | Decisive score |
| --- | ---: | ---: | ---: | ---: |
| Continuation off | 2-155-43 | 39.75% | 144 / 200 | 4.44% |
| Confidence-gated v4 | 11-153-36 | 43.75% | 148 / 200 | 23.40% |
| Risk-gated v5 | 15-145-40 | 43.75% | 140 / 200 | 27.27% |

### V5 versus continuation off

| Guardrail | Estimate | Confidence interval | Result |
| --- | ---: | ---: | --- |
| Paired score difference | +4.00 pp | two-sided 95% −0.50 to +8.50 pp | Fail |
| Avoidable-threefold difference | −2.00 pp | one-sided 95% −9.50 to +5.50 pp | Fail |
| Win-rate difference | +6.50 pp | one-sided 95% +3.50 to +10.00 pp | Pass |
| Decisive-score difference | +22.83 pp | one-sided 95% +11.50 to +34.23 pp | Pass |

### V5 versus v4

V5 changed score by 0.00 pp (95% interval −4.50 to +4.50 pp), reduced the
avoidable-threefold point estimate by 4.00 pp, increased win rate by 2.00 pp, and
increased decisive conditional score by 3.87 pp. These comparisons show no point
regression in the requested strength or decisive metrics, but they do not establish a
strength gain over v4.

## Dashboard and artifacts

The dashboard is live at `http://127.0.0.1:8765/`. It reports the aggregate 200-game
15-145-40 arena, 140 avoidable-threefold games, fixed-compute `max_steps` stop reason,
restored step 190, and the explicit state `strength/repetition uncertainty · no
generation`.

- Primary gate: `artifacts/evaluations/ekim-v5-paired-gate.json`
- V5/v4 comparison: `artifacts/evaluations/ekim-v5-vs-v4-paired.json`
- Candidate: `artifacts/ablations/ekim-risk-gated-20260825-01/`
- Audit: `artifacts/audits/ekim-repetition-risk-20260825-01/`

Generated training state remains excluded from Git by design. The failed candidate is
not published as a champion release.

## System and verification

The run used the local user-reported 32-GPU-core Apple M4 Max variant with 14 CPU
cores and 36 GiB unified memory. MLX reported `applegpu_g16s`, a 30.15 GB recommended
working-set limit, and macOS 26.4 arm64 in the persisted result artifact.

- `.venv/bin/ruff check .`: passed
- `.venv/bin/pytest -q`: 134 passed before and after the run
- `npm --prefix dashboard-ui run lint`: passed
- Dashboard health and `/api/snapshot`: passed after restart
- New generation: not started
- Champion: unchanged
