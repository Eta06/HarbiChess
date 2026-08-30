# YELKEN stable-base plus plastic-residual result

## Decision

The stable-base/plastic-residual architecture is implemented and function
preserving, but no preregistered cached arm passed every exact cumulative gate.
No fresh rolling-replay pilot was started. Production continuous learning,
generation, and promotion remain blocked.

## Implementation

- `HarbiChessPlasticValueNetwork` keeps the qualified MIHVER value path intact
  and adds a separate global/invariant plus spatial value tower.
- The plastic output is zero initialized. Measured initial maximum absolute
  policy-logit delta and WDL-logit delta were both exactly `0.0`.
- Stable mode trains only the new plastic value parameters; the existing DEVRIYE
  policy behavior is not changed by the value ablations.
- The audit records cached replay provenance, immutable-parameter hashes, old and
  fresh WDL metrics, gradient geometry, continuation ranking, tactical retention,
  and dashboard stop state.

## Frozen base-plasticity matrix

Artifact: `artifacts/runs/yelken-stable-plastic-ablation-20260830-01/result.json`

| Arm | Result | Old micro CE at step 1 | Old Pearson at step 1 | Fresh micro CE at step 1 |
|---|---:|---:|---:|---:|
| frozen base | fail | 0.913053 | 0.452570 | 0.879213 |
| base at 0.1x | fail | 0.913980 | 0.452076 | 0.876499 |
| mutable base | fail | 0.913981 | 0.452075 | 0.876497 |

MIHVER baselines were old micro CE `0.912848`, old macro CE `0.937718`, old
Pearson `0.452612`; fresh micro CE `0.879576`, fresh macro CE `0.879562`, and
fresh Pearson `0.600681`. Freezing MIHVER was the least destructive arm, while
making the base more plastic increased old-domain drift. No arm reached the
continuation/tactical gate because none passed the numeric gate.

## Gradient-geometry audit

Artifact: `artifacts/runs/yelken-constrained-plastic-ablation-20260830-01/result.json`

- The first historical/fresh gradient cosine was `+0.0671`, but observed cosine
  fell as low as approximately `-0.956` during training.
- PCGrad behaved like the mean while gradients initially agreed and did not pass.
- Two-objective MGDA assigned `89.29%` of the first update to historical replay,
  then reached `100%` historical weight by step 40. It improved old micro CE to
  `0.906689`, but old macro CE (`0.942753`) and Pearson (`0.451075`) still
  regressed. This rejects optimizer geometry alone as sufficient.

## Target-conflict audit

Artifact: `artifacts/runs/yelken-value-target-conflict-20260830-01/result.json`

- Historical pool: 196 games / 32,291 labelled positions.
- Fresh DEVRIYE pool: 88 games / 4,121 labelled positions.
- 101 exact encoded states occur in both pools; 51 carry conflicting WDL
  outcomes. Their mean merged target entropy is `1.0186` bits.
- The conflict is real at trajectory, board, and encoded-history identity levels,
  rather than a FEN-normalization artifact.

This is expected Monte Carlo outcome variance, but one-hot labels make exact
cross-domain Pareto constraints unnecessarily noisy at this scale.

## Uncertainty-preserving target ablation

Artifact: `artifacts/runs/yelken-soft-wdl-ablation-20260830-01/result.json`

- Fit-only aggregation found 74 ambiguous exact states spanning 404 rows and
  replaced their contradictory one-hot outcomes with one shared empirical soft
  WDL distribution. Validation labels stayed untouched.
- Six-objective normalized MGDA separately represented old/fresh micro CE, macro
  CE, and Pearson. The soft-MGDA arm was the least destructive.
- At its first step, soft-MGDA moved old micro CE from `0.91284848` to
  `0.91287669` (`+0.00002821`) and old Pearson from `0.45261201` to
  `0.45258009` (`-0.00003192`), while fresh micro CE improved from `0.87957586`
  to `0.87944497` and fresh Pearson improved to `0.60077720`.
- Those old-domain changes are tiny, but the frozen gate required literal
  no-regression. The arm therefore failed; thresholds were not changed after the
  result.

## Interpretation and next decision

The requested architecture solved catastrophic forgetting mechanically: MIHVER
remained byte-stable and all learning lived in a separate residual. It did not
make finite-sample Monte Carlo outcome distributions jointly Pareto-improvable.
Adding more base learning, steps, optimizer tricks, or another residual-size
sweep is not supported by the evidence.

The next scientifically defensible experiment is not to retroactively pass these
runs. It is to preregister a fresh, game-paired statistical non-inferiority gate:

1. retain the stable MIHVER base and soft repeated-state targets;
2. use a fixed, untouched stability set and paired game bootstrap confidence
   intervals instead of requiring every noisy point estimate to improve by a
   floating-point amount;
3. still require a positive lower confidence bound for fresh WDL/calibration or
   continuation improvement, plus unchanged Full Gumbel tactical strength;
4. increase independent terminal-game evidence only to a sample size fixed by a
   power calculation, not until a preferred answer appears.

That is a gate redesign for a new fresh experiment, not a relaxation or
reinterpretation of the failed YELKEN evidence.

## Verification

- Full test suite: `389 passed in 2.90s`.
- Ruff: all checks passed.
- Dashboard final state: `FAILED`, reason `soft_wdl_gate`, fresh generation
  blocked, promotion not ready.
- YELKEN artifacts occupy approximately 1.1 MiB in total.
