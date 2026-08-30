# MIHVER value representation result

## Decision

The policy-preserving nonlinear invariant value representation passes every frozen gate. The value
head is no longer a random-baseline-equivalent blocker:

- deterministic material: MSE `0.000126`, MAE `0.00857`, Pearson `0.99854`;
- held-out WDL: micro CE `1.09968 -> 0.91285`, macro CE `1.09890 -> 0.93772`, Brier
  `0.66745 -> 0.56387`, expected-score Pearson `0.0153 -> 0.4526`;
- outcome means: loss `-0.2340`, draw `-0.00683`, win `+0.2178`;
- continuation ranking: mean Spearman `-0.04234 -> +0.07316` (`+0.11550`), verified-top
  agreement `28.125% -> 37.5%`;
- Full Gumbel 256 tactical: `4/8 -> 5/8`, with no baseline-solved case lost;
- full validation policy logit maximum absolute delta: exactly `0.0`.

Auxiliary material predictions and all frozen release parameters remained exact. The selected WDL
checkpoint is `global-wdl`, SHA-256
`6c535585e952b3d8ba5ff2331fe4dc992d94c586fdfa6a7c3c0cbc9e2d229dbb`.

This resolves the value-representation qualification blocker. It authorizes designing the next
continuous-learner integration stage, but does not itself authorize a generation or promotion. The
dashboard therefore remains idle/passed with `promotion_ready=false`.

## Causal audit

1. Averaging binary piece planes hid count magnitude. Count-scaled current-board invariants fixed
   the deterministic material probe.
2. A single WDL distribution was an inappropriate material-probe target: scalar material accuracy
   and draw-logit calibration produced conflicting gradients.
3. Decoupling deterministic material supervision from production WDL preserved the probe while
   allowing calibrated outcome learning.
4. Outcome-only sampling favored decisive separation; mixed natural/outcome sampling fixed micro
   calibration but a linear WDL projection still missed the macro gate.
5. A zero-output 64-unit nonlinear invariant head represented phase/material interactions without
   changing initial policy or value behavior. This was the first architecture to pass both WDL gates.
6. The legacy continuation diagnostic bypassed residual value heads through the private release-only
   path. It now evaluates the complete public WDL forward path.

## Frozen artifacts

- Passing WDL result: `artifacts/runs/mihver-nonlinear-wdl-20260830-01/result.json`
- Passing downstream result: `artifacts/runs/mihver-value-downstream-20260830-01/result.json`
- Failed linear mixed-sampling control:
  `artifacts/runs/mihver-mixed-wdl-20260830-01/result.json`
- Dashboard: `artifacts/dashboard/state.json`

## Commit and file manifest

| Commit | File |
|---|---|
| `a29665c` | `docs/runs/mihver-invariant-value-preregistration-20260830.md` |
| `f086218` | `src/harbichess/backends/invariant_value_network.py` |
| `f0533db` | `tests/test_invariant_value_network.py` |
| `e77bccf` | `src/harbichess/evaluation/invariant_value_probe.py` |
| `6523536` | `tests/test_invariant_value_probe.py` |
| `656ff0f` | `docs/runs/mihver-count-scaled-value-preregistration-20260830.md` |
| `cd50669` | `src/harbichess/backends/invariant_value_network.py` |
| `1f8d478` | `tests/test_invariant_value_network.py` |
| `27f5348` | `docs/runs/mihver-wdl-calibration-preregistration-20260830.md` |
| `eacf4fa` | `src/harbichess/backends/invariant_value_network.py` |
| `58a804f` | `tests/test_invariant_value_network.py` |
| `d335e0e` | `src/harbichess/training/invariant_wdl_transfer.py` |
| `7158389` | `tests/test_invariant_wdl_transfer.py` |
| `3aa91ad` | `docs/runs/mihver-distributional-material-preregistration-20260830.md` |
| `19cf080` | `src/harbichess/evaluation/invariant_value_probe.py` |
| `9181354` | `tests/test_invariant_value_probe.py` |
| `e62fc64` | `docs/runs/mihver-balanced-material-preregistration-20260830.md` |
| `465ee86` | `src/harbichess/evaluation/invariant_value_probe.py` |
| `802b073` | `tests/test_invariant_value_probe.py` |
| `09276a3` | `docs/runs/mihver-decoupled-value-heads-preregistration-20260830.md` |
| `c4d066d` | `src/harbichess/backends/decoupled_value_network.py` |
| `46e832d` | `tests/test_decoupled_value_network.py` |
| `cce36ee` | `src/harbichess/training/decoupled_value_transfer.py` |
| `7dc519f` | `tests/test_decoupled_value_transfer.py` |
| `5571793` | `docs/runs/mihver-mixed-WDL-sampling-preregistration-20260830.md` |
| `fd93447` | `src/harbichess/training/decoupled_value_transfer.py` |
| `f0ed8a5` | `tests/test_decoupled_value_transfer.py` |
| `7e7d59d` | `docs/runs/mihver-nonlinear-invariant-preregistration-20260830.md` |
| `4565048` | `src/harbichess/backends/decoupled_value_network.py` |
| `74b1e48` | `src/harbichess/training/decoupled_value_transfer.py` |
| `0ff222f` | `tests/test_decoupled_value_network.py` |
| `7b132ea` | `docs/runs/mihver-value-downstream-preregistration-20260830.md` |
| `4fc3020` | `src/harbichess/training/joint_policy_value_transfer.py` |
| `6d3c8b9` | `tests/test_joint_policy_value_transfer.py` |
| `18be110` | `src/harbichess/evaluation/decoupled_value_qualification.py` |
| `de8a303` | `tests/test_decoupled_value_qualification.py` |
| `a006320` | `src/harbichess/evaluation/decoupled_value_qualification.py` |
