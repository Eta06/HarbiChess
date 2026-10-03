# AYNA pairwise policy transfer: before implementation/training

The all-parameter dense-policy bootstrap failed held-out qualification. Keep it
failed and preserve every checkpoint. Its step-200 WDL improved; select that
value state using **validation only**, never arena. Policy has 1.20M of 1.26M
parameters, trained from 2605 examples, with subsequent severe held-out overfit.

Hypothesis: a shared scorer conditioned on origin, destination, current pieces,
board coordinates and global counts can generalize move preference with about
20k trainable parameters. This is a conventional relational/attention mechanism,
not a claimed original invention. Unlike failed YAPI's origin-only adapter,
destination content and origin/destination interaction are explicit. Unlike prior
tiny frozen adapters, direct current-piece and whole-board features are exposed.

Freeze encoder/action/search semantics. Add versioned `pairwise` architecture:
same inherited decoupled trunk/value; remove dense policy; each square has a
64-unit ReLU embedding of trunk+12 current-piece planes+20 normalized global
features+2 coordinates. Query/key width32 dot product /sqrt32, plus learned
origin-plane and destination-plane scores. Canonical action destination lookup
handles queen rays, knights, castling, en-passant and underpromotion geometry;
off-board actions have -1e9 logits and legal support is still rule-generated.
No teacher at inference. Torch and real MLX CPU forward/masked/soft-target
loss/gradient parity must pass; Apple Metal remains untested.

Explicit **weights-only architecture transfer** from dense step200 SHA
180f2f8fc5daf768b568ac1c62594bd213b61cbdf920c556655f06b5e97d4c09:
copy trunk/value tensors exactly, reset new policy, reset optimizer. Preserve
all source tensors/files. Weight schema1 contains new architecture spec;
old readers must reject unknown architecture. Legacy unversioned pairwise
loading is refused. New full checkpoints record trainable names and task config.

Reuse the identical train/validation dataset SHA
33640698d2462894376b48d7afbed76ddd6b3a20544bd543985ed395690c3cd4,
openings SHA 019c5b4a49ac287ca03effb65b33ed97daa2e702ddd4a6972491969e27912b24.
This validation set has already guided an architecture hypothesis, so it is a
development holdout, not independent final evidence. All 32 arena families
remain unseen, and are not used for checkpoint selection.

Seed20261003, only `pair_` parameters trainable, one CPU Torch thread, AdamW
LR2e-4/wd1e-4/clip5, batch64, 6000 updates/1800seconds, evaluate every200,
early stop1000 without total-CE improvement >=1e-4. Initial and inherited value
metrics must be identical. Qualification requires policy CE <=3.172551
(original dense initial minus0.10) and unchanged value CE <=0.776855 (original
minus0.10), with finite parameters and legal normalized outputs. All other
bootstrap/arena uncertainty, colour pairing and cap rules remain frozen.

If qualified, run the original untouched 32-family/64-game arena vs dense
initial and Stockfish32, and the dense initial vs Stockfish32 control. This is
the first use of that arena. Score>0.60 vs initial, paired bootstrap lower>0.50,
caps<=10% required for development strength improvement, not high-level Elo.
Report size, full/masked latency and actual game throughput; a smaller model
is not automatically faster. Failure remains failed; no repeated arena selection.
