# KRITIK value-collapse audit result

## Decision

The frozen joint policy+value transfer is **not qualified**. Continuous learning, generation,
arena, and promotion remain blocked. Policy imitation is not treated as success.

The evidence rejects three narrower explanations:

1. **Training duration:** later checkpoints improve train metrics while held-out metrics degrade.
2. **Only too few independent games:** expanding from 43 to 148 train trajectories strengthens
   train separation but not held-out outcome separation.
3. **Only equal loss weights:** measured gradient balancing does not recover held-out WDL or policy
   transfer.

The current blocker is value representation and generalization under the available data regime.

## Corrected replay scale control

Artifact: `artifacts/diagnostics/kritik-corrected-replay-value-20260830-01/result.json`

- Eight compatible schema 10-12 replay runs all use baseline SHA-256 `5a094285...`.
- 384 named games, 360 unique trajectories, 24 duplicate trajectories.
- 196 unique known terminal trajectories; 188 unknown/max-ply games excluded from WDL.
- Trajectory-disjoint split: 148 train / 48 validation, zero fingerprint overlap.
- Best checkpoint: step 20.
- Validation macro WDL CE: `1.09890 -> 1.09743`.
- Validation Pearson: `0.0153 -> 0.0492`.
- Validation loss/draw and draw/win margins: `0.00096 / 0.00144`.
- At step 400, train Pearson reaches `0.2173` and margins `0.0663 / 0.0603`, while validation
  Pearson is `-0.0237`. This is train-trajectory fitting, not value generalization.

Frozen gate result: **failed**.

## Shared-representation joint transfer

Artifact: `artifacts/runs/kritik-shared-representation-joint-20260830-01/result.json`

- Starts the shared trunk, policy head, and WDL head from the release baseline.
- Qualified Full Gumbel targets: 384 train / 192 validation positions.
- Corrected schema-12 terminal WDL rows: 7,698 train / 2,340 validation.
- Policy train CE improves strongly (`2.895 -> 1.586` by step 240), while policy validation CE
  degrades (`2.871 -> 3.988`).
- WDL validation briefly reaches Pearson `0.228` at step 240, but outcome margins remain only
  `0.0041 / 0.0046`, macro CE improves only `0.0038`, and policy validation has already regressed.
- No checkpoint passes policy and WDL gates together. Continuation ranking, tactical suite, and
  arena therefore do not run.

Frozen gate result: **failed**.

## Loss-gradient audit and fixed balancing control

At the release baseline on the deterministic first joint batches:

- policy gradient norm on shared trunk: `0.105641`;
- WDL gradient norm on shared trunk: `0.024442`;
- policy/value norm ratio: `4.3221`;
- policy/value trunk-gradient cosine: `0.0668`.

The preregistered `policy=0.25`, `value=1.0` control is recorded in
`artifacts/runs/kritik-gradient-balanced-joint-20260830-01/result.json`.

- Best joint checkpoint remains step 20 and fails every WDL magnitude/separation gate.
- Validation WDL at step 140 reverses outcome order (Pearson `-0.186`, margins
  `-0.0314 / -0.0925`).
- Policy validation CE also degrades (`2.871 -> 3.412`).

Frozen gate result: **failed**. No further scalar loss-weight sweep is justified.

## Deterministic value-representation probe

Artifact: `artifacts/diagnostics/kritik-deterministic-value-probe-20260830-01/result.json`

This removes game outcomes and future variance entirely. The target is deterministic depth-0
side-to-move material value, which is directly recoverable from the current encoded piece planes.
The split remains trajectory-disjoint and uses 8,192 train / 4,096 validation positions.

| Arm | Best step | Validation MSE | MAE | Pearson | Gate |
|---|---:|---:|---:|---:|---|
| Release baseline | 0 | 0.02335 | 0.11504 | -0.0287 | reference |
| Value head only | 40 | 0.02319 | 0.11222 | 0.0153 | failed |
| Full representation | 20 | 0.02293 | 0.11209 | 0.1115 | failed |

Even the full trunk cannot generalize this simple invariant function under the frozen schedule.
The current flatten-based value path is therefore a structural small-data bottleneck, not merely a
bad terminal label or insufficient training time.

## Next highest-probability correction

Do not change the search allocator and do not generate more replay yet. The next experiment should
be a function-preserving policy network with a new value representation that explicitly exposes
global/invariant summaries (at minimum pooled current piece planes plus pooled trunk features) to
the WDL head. First qualify it on the deterministic probe. Then train it with an auxiliary
deterministic/short-horizon value objective plus corrected terminal WDL while preserving Full
Gumbel policy outputs. Only after deterministic value, held-out WDL, calibration, continuation
ranking, and `4/8` tactical retention pass should joint transfer or continuous policy iteration be
reopened.

