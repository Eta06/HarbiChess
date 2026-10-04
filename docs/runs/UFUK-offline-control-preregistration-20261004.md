# UFUK: separately test retained offline self-target control

New hypothesis after width and Q-range failures: the previously retained
head-only offline self-target control may improve playing strength despite the
failed additional-adapter comparison. This is explicitly after reviewing old
loss results, not a retroactive success of that representation experiment.
Old protocol/result remain failed and unchanged; no adapter promoted or trained
longer. The new test selects only the original heldout-selected control8000,
not a new checkpoint scan or post-game choice.

Candidate SHAef2ee1d26bb9b0e80a9e849009c7ed326b43afe9a9881f0160ad2cb14a6dc96a,
original18f comparison. Actual old clean training source59dde8d,8000updates/
512000sampled own-search rows. Fixed96own games/72train/24heldout, search-derived
policy targets only; head16914parameters trainable, inherited55583bitwise frozen.
Original precommitted candidate losses passed own-target gain>=0.05 (0.076538)
and native-policy retention<=0.15 (+0.112699), exact unchanged native value/Q.
Those observations justify this new model-only game hypothesis; they are not
already a strength result. No new gradients or teacher labels for this test.

Before games freeze24new12ply legal-stress openings with seed20261019, uniform
legal moves from the initial board, no model/engine/outcome-quality filter.
Require unique four-ply histories AND four-ply FEN keys outside all available
native/current self-replay stage inputs, frozen opening books and prior complete
arenas; current roots absent from80511native labels and prior complete arenas.
Merge duplicate four-ply position keys rather than call transpositions independent.
These are new stage families, not a typical natural-opening distribution.
Inherited historical model pretraining exposure may be incomplete; do not claim
universal independent pretraining holdout or assign general engine Elo.

Both models identical16sim/max4/Gumbel0/value_scale0.1/maxvisit_init50,
CPU1FP32/eagerB1;48colour-matched games per arm, max400plies/900s per arm,
seed20261019/claim_draw=True. Arms:candidate-vs-initial and each-vs-Stockfish512.
SF19/Threads1/Hash16MiB/512requested with actual per-ply nodes. This stronger
reference budget differs from oldSF32; no merged score/Elo or equal-compute claim.

Primary strength gate score>0.60,24whole-pair bootstrap95%lower>0.50 and
unknowncapfraction<=0.10; conservative Hoeffding interval beside conditional
bootstrap. Caps diagnostically0.5, never relabelled as real draws. Secondary
pairedSF512 delta and95% uncertainty; no lucky extra roots/seeds/updates.
Stop invalid/nonfinite/source/checksum/budget; preserve durable moves/incomplete
states. No promotion or next self-learning run merely from lower CE. Record
wall,NNcalls,median/p95 move timings,legal histories,colour pairing and full
checkpoint/frozen-state integrity. Existing4CPU/16GiB/noGPU/no paid allocation.

Even a pass means bounded offline self-produced policy distillation, not fresh
closed-loop progress, Stockfish/AlphaZero level, teacher-system superiority or
novel algorithm. A fresh policy-iteration follow-up would need its own protocol
and final-versus-initial same-search gate. Previous failures remain failed.
