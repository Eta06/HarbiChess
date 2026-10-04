# UFUK: prior-directed RL primary code inspection

Public primary repository pinned at
[ac7d5c622c001a96387be5a773ddec1876adbff9](https://github.com/szmilosz/prior-directed-exploration/tree/ac7d5c622c001a96387be5a773ddec1876adbff9),
commit31August2026, read4October2026. README4882bytes/SHA370fb1f93fa0ab5d65d472bf2bf75596dd3327a104db5473bf22346fa67a1212;
train.py14245/SHAaf9fbc18562c7de3ffb489bb7b825da35a3cf91dbbaf02131d36e197bfe15946;
selfplay.py16560/SHAfbc0960e9931bff22db21e5a2e4605a0f11d9ea2f62020b6b1600316cfff335d.
MIT LICENSE also saved. [Exact source/read receipts](UFUK-prior-directed-code-evidence-20261004.json).
This is source inspection, NOT execution, reproduced training or our result.
No network/assets installed/downloaded beyond these bounded text sources; no
external tests run with missing-network skips and then called successful.

The repository uses three copies of released191Mparameter BT4 Chessformer,
strong supervised/search-distilled baseline,1024parallelgames/2000steps. Its
reported modest self-learning/puzzle gains do not transfer to our weak72k/303k
models or imply small-prior RL will improve. Full hardware/ratings/selection
limitations remain in [prior primary-paper analysis](UFUK-efficient-selfplay-primary-20261004.md).

Actual `step_loss`: detached sampled-action advantage times cross-entropy and
truncated importance ratio; forwardKL(base-policy || current-policy) implemented
by KL-div on current log-softmax; separate STM WDL target KL plus frozen-base
WDL anchoring. Native expected score is W+0.5D, so the corresponding Q=2E-1 is
W-L. `wdl_to_pov` swaps W/L when mover changes, retaining D. Live probability
targets/advantage/importance are treated as constants for policy gradients.
One fresh transition batch per optimizer step, minibatch gradient accumulation,
then EMA; no replay extra epochs. These are conventional mechanisms, not ours.

Three reproducibility details matter before adaptation:

- Paper2608.27757v1 Table2 says max512plies "drawn at cap". In this pinned code,
  `sample_transitions` restarts at `game.plies>=max_plies` but preserves result
  `"*"` if the board is actually ongoing. `label_transitions` then uses flipped
  EMA bootstrap for that ongoing successor, not a draw one-hot. Thus the inspected
  released snapshot's cutoff supervision differs from the paper table. This
  does NOT prove which code/config generated published checkpoints; reproduction
  provenance is unresolved. HarbiChess must continue to distinguish unknown
  cutoff, true terminal outcome and explicitly labelled bootstrapped estimate.
- The CLI defaults entropy temperature's WDL source to ONLINE; EMA is an explicit
  selectable option, adding another forward pass. The paper describes EMA-derived
  uncertainty. An adaptation must register which one plus measured extra cost,
  not assume code defaults reproduce a particular reported run. WDL entropy is
  uncertainty, not proof of tactical criticality or correct value calibration.
- `save_checkpoint` stores model/optimizer/step/base/EMA weights but not active
  1024game full histories, Python/NumPy/Torch random states, immutable input
  hashes/runtime or sampler/rollout cursor. That payload alone is not exact full
  ongoing self-play resume. Our future online loop would need all those states,
  explicit versions and a fresh-process next-update/trajectory test. Current
  supervised native model/Adam/RNG/index-trace/source/cache resume already has
  stronger within-stage integrity; it does not automatically cover a new online
  stateful algorithm or full cross-framework migration.

Draw convention also differs: code claims only CURRENTthreefold/50move conditions
via `is_repetition(3)`/`is_fifty_moves`, plus automatic python-chess terminals.
HarbiChess's declared `claim_draw=True` convention can include a claim by an
announced next move. Both are concrete conventions; silently switching them
would change trajectories, training targets and evaluation comparability.
Old replay/checkpoint semantics and failed experiments must remain intact.

Decisions: preserve current prospective capacity controls unchanged. Do not
import this191Mbackbone, its checkpoints/data or hyperparameters as our main
solution. If a later measured stronger prior supports a controlled KL/TD/EMA
self-learning experiment, implement an explicit HarbiChess version with same
legal104history/4672action/STM contracts, persistent active games/RNG/base/EMA/
optimizer state, unknown-cap handling, unchanged-control compute and independent
initial/final strength gates. Neither this inspection nor the ongoing teacher
capacity experiment establishes self-learning, Stockfish level or novelty.
