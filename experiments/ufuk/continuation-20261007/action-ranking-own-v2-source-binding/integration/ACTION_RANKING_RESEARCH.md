# Selected-action ranking: signal and limits

## What the sealed rows say

The sealed ownQ-v2 conversion exposes 1,024 actually played `selected_best_uci`
actions, one completed root mover-Q scalar per row, and complete legal history.
The target rows do not contain an outcome or search score for every other
legal action. The new afterstate conversion changes the viewpoint of that
same scalar after the selected move; it does not add counterfactual evidence.

Therefore:

- A value-derived ranking can score every legal move with
  `logit(a) = -V(child(s,a))/T`, then use one-hot cross entropy on the logged
  selected-best move. Sibling moves are candidate classes in that softmax;
  they are not labeled losing outcomes. This is **search-policy imitation**,
  not action-outcome learning. It may amortize the current search's choice,
  while copying the same parent's errors and discarding visit/value
  uncertainty.
- Assigning `-1`, draw, or a losing target to every unselected legal move is
  unsupported. Those are unsearched counterfactuals, not labeled negatives.
- A ranking loss needs pairwise Q/visit targets for alternatives. The current
  rows do not provide them; collecting them requires new parent search and a
  fixed extra compute budget, or a prospectively fixed redistribution of the
  existing root-search budget. It is a separate data phase and must retain
  full-history/protected-alias checks.

## Closest methods

- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed.,
  §6.8: afterstates and temporal-difference learning. Moving a return target
  from a state/action pair to the resulting state is established viewpoint
  bookkeeping, not a new learning algorithm.
- Anthony, Tian, and Barber, “Thinking Fast and Slow with Deep Learning and
  Tree Search,” NeurIPS 2017, [arXiv:1705.08439](https://arxiv.org/abs/1705.08439):
  Expert Iteration alternates search-generated improvement targets and a
  learned policy/value. A one-hot label from this run's current search is a
  weaker form of that standard self-distillation loop.
- Silver et al., “Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm,” 2017,
  [arXiv:1712.01815](https://arxiv.org/abs/1712.01815): AlphaZero uses MCTS
  visit distributions for policy supervision and completed self-play outcomes
  for value. The available ownQ rows lack the full visit distribution.
- Danihelka et al., “Policy Improvement by Planning with Gumbel,”
  [arXiv:2112.00178](https://arxiv.org/abs/2112.00178): search-policy
  improvement depends on explicit root action-value/visit information; it
  does not justify labeling every legal alternative as a loss.

## A bounded future comparison

If the existing ownQ candidate fails its frozen gate, a bounded ranking
ablation can use the same 1,024 roots. For every legal move, compute the
current parent value of its child and form `logit(a) = -V(child)/T`; train
one-hot cross entropy on the logged selected-best move. Keep the existing
selected-child Q regression as an anchor and add `0.1 * CE`, with `T=0.25`
fixed before the run. These constants are a proposed starting protocol, not
results or tuned values. Unselected children are only classes in the softmax;
they receive no fabricated Q or WDL target. This tests whether a ranking
signal helps while retaining the scalar target that is actually observed.

The anchor constrains the selected action's scalar value, but cannot tell
whether another legal move was equally good. One-hot supervision still
prefers the logged action over every other class. The 1,024-root legal-action
support is therefore broader than one selected-move target, but only covers
those source positions and supplies no sibling outcome data. This is the main
noise risk, not a missing terminal label that can be safely filled in.

Keep the same parent initialization, data, total update budget, full-resume
proof, and unchanged two-seed arena gate. Call it “own-search action
imitation,” and compare it against the current value-only afterstate
candidate. It tests whether amortizing the parent's selected move helps the
same search; it does not establish independent policy improvement.

The expected gain is uncertain. A sharp softmax may copy search mistakes;
one-hot labels treat several equally good moves as alternatives to suppress;
and the CE can pull the value-derived ranking against the anchored Q target.
Evaluating every legal child adds roughly one forward pass per legal move
unless the evaluator batches efficiently. These are primary risks to measure
before allocating a full two-seed fit.

Do not include unsearched-action value negatives. A stronger ExIt/Q-ranking
test would need fresh root packets with search evidence for alternatives,
predeclared action coverage and compute, and a separate immutably sealed
dataset. Neither alternative scores nor training results were generated for
this note.
