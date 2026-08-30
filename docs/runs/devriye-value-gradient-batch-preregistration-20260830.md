# DEVRIYE value gradient batch ablation preregistration

## Prior result

The cached output-distillation matrix (`0`, `0.25`, `1`, `4`, `16`) produced no
passing arm. At the initial weights the distillation KL and its gradient are
zero, so every one-step arm was identical. The unanchored update improved old
micro CE and Pearson but regressed old macro CE; it also regressed fresh
game-disjoint CE from 0.893285 to 0.896311. Output anchoring is rejected.

## Frozen hypothesis and matrix

The selected value checkpoint is one optimizer step, but the current step sees
only 32 historical and 32 fresh positions. Those fresh rows come from a small
subset of terminal games, making the gradient estimator noisy even when the
replay generation contains enough independent games.

- Reuse the exact cached `-13` update-1 fit/held-out game split and MIHVER start.
- Run exactly one Adam update at learning rate `1e-4`.
- Test total value batch sizes `64`, `256`, `1024`, and `2048`.
- Keep the batch half historical and half fresh.
- Fresh halves retain the fixed 25% loss / 50% draw / 25% win allocation and
  sample games uniformly inside outcome. Historical halves retain the existing
  mixed sampler.
- No policy update, anchor loss, extra step, or target change is allowed.

## Gate

An arm passes only if old fixed-validation micro CE, macro CE, and Pearson do
not regress, while fresh game-disjoint CE improves and fresh Pearson does not
regress. Outcome margins and finite-gradient protection remain mandatory.

If multiple arms pass, choose the smallest. If none passes, larger batches are
rejected and the value target/objective must be redesigned.
