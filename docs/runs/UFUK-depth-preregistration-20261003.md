# UFUK representation depth: before training

Teacher bootstrap improved development strength but two genuine self-learning
phases failed; disabling search also failed. Isolate representation depth before
another self-learning run. This is teacher-assisted capacity testing, not pure
self-learning, novel architecture or a claim of Stockfish-level play.

## Fixed initial function and control

Both arms start from the same native-bootstrap step9250 weights,
SHA`18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae`.
Control unchanged72,497-parameter pairwise model,16channels/2residual blocks in
both spatial policy and value towers. Treatment adds2residual blocks **per tower**,
same widths and all heads;91,057 parameters. All inherited tensors identical.
New blocks have random first convolution and zero final convolution/bias, so
their initial residual branch is exactly zero on nonnegative ReLU features.
This standard function-preserving deepening is not an originality claim.

Version1 weights-only architecture transfer, seed20261009, optimizer reset for
both; explicitly not old training resume. Require trained model full/masked
logits bitwise equal in Torch before training on six rule-special FENs plus
history positions, real MLX CPU full/masked numerical parity2e-5. New residual
branches must receive gradients; zero-weight branches may not conceal testing.
Preserve original model. Record initial hashes/specifications/transfer receipt.

Complete weight loading formerly consumed Torch sampling RNG through discarded
constructor initialization. The new loader preserves CPU RNG for all five
architectures; this makes both new arms use the same batch index stream. Legacy
full checkpoint restores its saved RNG after load; demonstrate one genuine
archived optimizer update matches pinned pre-fix source, without changing the
original checkpoint. Existing historical runs retain their actual source/hash
and sampler states; do not rewrite their provenance or expected samples.

## Same data and learning budget

Frozen native dataset1,024games/80,511rows,manifestSHA
`b34a3870b494b4cc0fe8a1df87960fe5e2406a5c9a2ed341d1b93fed0e0e094f`.
Same family train/validation splits and caps80,000train/20,000validation;
seed20261009 means both validation subsets identical, but differs from prior
seed20261005 cached20,000; evaluate frozen reference directly on new subset.
All parameters/AdamWLR5e-5/wd1e-4/clip5/batch64,CPUFP32/one thread/deterministic.
10,000update ceiling,eval/checkpoint every250,early stop1,500updates since best
total policy+value CE improvement≥1e-4. Same sample stream at equal cursors;
compare saved RNG at common checkpoints. Single seed, no multi-seed inference.

Run arms sequentially within4CPU/16GiB,2400s **total wall cap per arm including
both process setups**. End at250updates in a fresh process, full-checkpoint resume
to10,000 with remaining wall cap. No extra continuation to reach favourable loss.
Stop on malformed/nonfinite/checksum/resource/protocol failure, preserve partial
data and last immutable checkpoint. If cap reached, report paused/stopped at
budget, not completed10,000. No paid resources or new teacher collection.

## Pre-arena decision and strength scope

Select best checkpoint by validation totalCE only, never arena score. Treatment
capacity gate requires policyCE at least0.03 below continued-small best and0.05
below frozen-reference CE, valueCE no more than0.03 above reference,QMAE no more
than0.02 above continued-small,top16 coverage no more than0.01 below continued-small.
Apply this external registered gate; the older trainer's built-in0.10/both-heads
bootstrap flag answers a different question and will also be reported unchanged.
Gate failure preserves control/reference and stops this capacity phase without
an arena or extending training. Passing permits a separately frozen matched
development arena before any inference default, promotion or self-learning change.

Record actual wall/CPU/peakRSS/updates/source/data/checkpoint hashes and test
results. Initial parity, fixed sampling and resume correctness do not prove
strength. AppleMetal/CUDA untested; actual Linux MLX CPU tests required, no skips.
Git and Space preserve earlier failures; Release400 still blocks remote binary
backup, local models/data must remain intact.
