# EYLUL v4 evidence gate — 2026-08-25

## Decision

The confidence-gated v4 candidate did not satisfy the pre-registered multi-part
evidence gate. It improved wins and decisive performance, but its paired strength
interval still narrowly included zero and avoidable-threefold behavior failed the
non-inferiority requirement. No new generation was started; the champion is unchanged.

Gate implementation source commit: `699ae53439f6c6751584f7c33806cf874e0081ab`.

## Pre-registered gate

The thresholds were implemented, tested, committed, and pushed before the new arena
games were played:

1. **Paired strength:** two-sided 95% bootstrap lower bound must be greater than zero.
2. **Avoidable-threefold:** candidate point estimate must not exceed control, and the
   one-sided 95% upper bound may be at most +5 percentage points.
3. **Win-rate guardrail:** candidate point estimate must exceed control and the
   one-sided 95% lower bound may be no worse than -2 percentage points.
4. **Decisive guardrail:** candidate conditional score among decisive games must not
   regress in point estimate.

The final decision requires all four conditions. Bootstrap used 50,000 deterministic
paired resamples with seed `2026082506`.

## Additional evidence

No model was retrained and continuation exposure was not changed. The persisted
continuation-off and confidence-gated v4 checkpoints each played 104 new games from
52 color-balanced opening pairs using independent seed `2026082572`. Combined with
the prior 96 matched games, the gate used 200 identical games per arm.

| Arm | W-D-L | Score | Elo | 95% Elo interval |
| --- | ---: | ---: | ---: | ---: |
| Continuation-off | 2-155-43 | 39.75% | -72.25 | -94.11 to -50.94 |
| Confidence-gated v4 | 11-153-36 | **43.75%** | **-43.66** | -66.85 to -20.84 |

## Gate result

| Evidence | Estimate | Confidence interval | Result |
| --- | ---: | ---: | --- |
| Paired score difference | +4.00 pp | -0.25 to +8.25 pp | **Fail** |
| Avoidable-threefold difference | +2.00 pp | one-sided -5.00 to +9.00 pp | **Fail** |
| Win-rate difference | +4.50 pp | one-sided +1.50 to +7.50 pp | Pass |
| Decisive-score difference | +18.96 pp | one-sided +7.55 to +30.77 pp | Pass |

The decisive result resolves one ambiguity from AGUSTOS: the directional score gain
is not solely losses becoming draws. The v4 candidate won 11 games versus only 2 for
off, and its decisive score was 23.40% versus 4.44%. Nevertheless, overall paired
strength evidence remains just below the required threshold.

Avoidable-threefold is the clearer blocker. Off produced 144 avoidable repetitions
in 200 games; gated produced 148. The +2 percentage-point regression is below the
allowed +5 margin in magnitude, but the pre-registered rule also requires the point
estimate not to worsen, and its one-sided uncertainty reaches +9 points. Relaxing
that condition after observing the result would invalidate the experiment.

## Next step

Do not advance a generation and do not increase training length or exposure. Preserve
schema v4 and the branch evidence artifacts. The next target change should explicitly
penalize or exclude qualified non-repeat policies that still increase the probability
of returning to an avoidable repeat later in the continuation. A short multi-ply
rollout-level repetition-risk field can distinguish a branch that leaves the immediate
repeat but loops back two or three moves later. Any subsequent candidate must rerun
this same frozen gate.

## Verification and artifacts

- `.venv/bin/ruff check .`: passed
- `.venv/bin/pytest -q`: 130 passed
- `npm --prefix dashboard-ui run lint`: passed
- Paired gate: `artifacts/evaluations/eylul-v4-paired-gate.json`
- Off evidence arena: `artifacts/ablations/temmuz-off-20260825-01/arena/eylul-off-evidence/`
- Gated evidence arena: `artifacts/ablations/agustos-gated-20260825-01/arena/eylul-gated-evidence/`
- Dashboard remained live at `http://127.0.0.1:8765/`
