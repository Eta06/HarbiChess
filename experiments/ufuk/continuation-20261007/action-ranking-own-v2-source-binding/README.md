# Own-search legal-child ranking proposal

This is a separate prospective phase over the same 1,024 eligible ownQ-v2
roots used by the frozen afterstate learner. It makes no new search calls and
uses no Stockfish labels. It is an action-imitation ablation, not an
independent policy-improvement claim.

For every original root, conversion replays the full history and builds all
legal child classes. A nonterminal child is included only if its canonical
piece-placement alias occurs in that root's original search trace. Missing
aliases fail conversion. Terminal children get exact WDL constants from the
child side-to-move and skip feature/value inference. Only the actually played
`selected_best_uci` is the one-hot class target. No sibling receives an
invented Q value or WDL outcome.

The fixed objective on a 16-root minibatch is:

```text
MSE(current selected-child value, existing afterstate Q target)
+ 0.1 * CE(softmax(-child_value / 0.25), selected_best_action)
```

The selected Q scalar is the anchor. Child values are from each child side to
move, so negating them ranks actions from the root mover's perspective. A
terminal selected child has no model Q prediction; its exact value is used in
the ranking logits and its Q-anchor term contributes zero. The run uses the
same-seed teacher model weights as a weights-only start, fresh Adam/RNG state,
and exactly 64 updates. It has a distinct data, phase, native, contract, proof,
and candidate schema from the afterstate-Q phase.

This is standard search-policy imitation in the Expert Iteration family. It
may distill the parent's selected move, but it can repeat the same parent's
mistakes. One-hot labels can suppress several equally good moves. Processing
all legal child classes costs approximately one batched sparse-value forward
per nonterminal legal child: 64 updates × 16 sampled roots × the mean legal
move count. The source search's aliases are reused as evidence of which child
inputs it evaluated; no child is rescored with a new engine or parent search
during conversion.

## Files

- `source/convert.py` replays the sealed afterstate conversion and original
  ownQ event/alias trace, creates every legal child class, and fails if a
  nonterminal child alias is missing from the original trace.
- `source/contract.py` validates move legality, full child histories, exact
  terminal values, selected target identity, trace membership, and the
  absence of sibling Q/WDL labels.
- `source/ranking.py` implements the fixed CE plus selected-child Q anchor.
- `source/native.py` stores the full model, immutable baseline, Adam, Python,
  Torch, and private sampler states for exact resume.
- `source/train.py`, `source/prove.py`, and `source/contract_builder.py`
  implement the CPU-only training phases, typed contracts, and proof.
- `integration/seal_factory.py` creates ROOT-clocked conversion/build/
  orchestration registrations from explicit request JSON. It never assigns
  clocks or launches work.
- `integration/ROOT_RUNBOOK.md` gives the prospective command sequence.
- `integration/ACTION_RANKING_RESEARCH.md` records the hypothesis and limits.

No real ownQ dataset conversion, model inference, search, training, or arena
run was performed while preparing this proposal. Synthetic tests exercise
data admission, target perspective, ranking gradients, and native resume.
