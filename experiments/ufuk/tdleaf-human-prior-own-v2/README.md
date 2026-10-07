# Human-prior TDLeaf own-learning candidate (source only)

This directory is a separate prospective lineage. It reuses the original
human-prior NNUE16 model and admitted current-parent weights, but collection,
targets, native state, phase, dataset, and proof schemas are distinct. No
production parent inference, self-play collection, conversion, SGD, or arena
run has occurred from this directory.

## Learning signal

The collector uses the PV-instrumented copy of the fixed 8192-node, qdepth-2,
max-depth-8 search. The completed search's selected move is played. Each row
retains the full played root history, selected move, full PV, exact leaf
history, root score, PV endpoint kind/value, mover, search counters, and the
complete evaluator-input alias trace. The v2 protected scope checks the current selected root, every actually played
state, every model input, and every state from the current root through its PV
leaf. A hit excludes the whole episode. Full ancestral prefixes are replayed
and archived with the exact protected-prefix intersections, but unqueried
ancestral states (including the shared standard-start board) are context, not
model exposure and not an unseen-history claim. The protected alias file and
4096-row CLASSIC-derived source distribution remain unchanged from the parent
registration. The v1 full-prefix rule is preserved as a separate blocked source
lineage; it was rejected before any actor collection because every root ancestry
contains the protected START alias.

The converter replays the game and PV under python-chess rules. It forms a
White-oriented finite-trajectory lambda return with lambda=0.5: exact WDL for
completed rule-terminal games, or the last logged pre-action root Q for a
game that ends at an UNKNOWN cap. UNKNOWN is never converted to a draw, and no
value is invented for the unseen post-action position. Static PV leaves receive
the target after the exact root/leaf parity transform. Terminal PV leaves are
verified against the rules and are omitted from the gradient set. Training
minimizes MSE over the remaining static-leaf rows, using the same prior-plus-
residual NNUE16, fresh Adam state, and fixed 64-update budget as the existing
human-prior own phase. This is a TDLeaf-style method, not a novelty claim.

The existing own-Q journal cannot supply leaf gradients: it did not preserve
the principal variation or leaf full-history state. This lineage therefore
requires new PV-bearing collection. Root-Q lambda targets and PV-leaf targets
must remain separately versioned.

## Prior art

- Baxter, Tridgell, and Weaver, [TDLeaf(lambda): Combining Temporal Difference
  Learning with Game-Tree Search](https://arxiv.org/abs/cs/9901001), 1999,
  §3/equations 8–9 and Figure 1. TDLeaf applies gradients at minimax-selected
  leaves; TD-directed instead learns from successive root evaluations.
- Lai, [Giraffe: Using Deep Reinforcement Learning to Play
  Chess](https://arxiv.org/abs/1509.01549), 2015, §4.5. It records successive
  search scores and applies lambda-weighted updates at the minimax leaf.

The cited work used substantially more experience/compute than this proposed
1024-row, 64-update path. It does not establish that this small parent-search
dataset can improve strength. A weak parent can select incorrect PVs and
reinforce its own errors; measured strength remains necessary.

## Scratch checks

Run in this directory, on CPU:

```sh
PYTHONDONTWRITEBYTECODE=1 /workspace/HarbiChess/.venv/bin/python -m pytest -q -p no:cacheprovider test_tdleaf.py
PYTHONDONTWRITEBYTECODE=1 /workspace/HarbiChess/.venv/bin/ruff check --config /workspace/HarbiChess/pyproject.toml .
```

These are synthetic tests only. Before any real use, this lineage still needs
an independently versioned six-packet audit that reruns and compares the exact
PV results/leaves, a sealed conversion/build contract, and the actual native
proof plus fresh-fit qualification. Existing own-Q/TD proofs do not qualify
this phase.
