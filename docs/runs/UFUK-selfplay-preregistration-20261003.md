# UFUK calibrated-value self-search policy iteration: before generation

The wider balanced end-to-end bootstrap passed its registered development arena:
20/26/2,score0.6875,pair bootstrap[0.604167,0.770833],24families/48games,no caps.
Its SF32 result1/13/34 remains weak; SF delta CI includes zero. Teacher imitation
is not self-learning. AYNA's earlier all-parameter self-learning failed its primary
and increased native reference QMAE. Preserve that failure; do not reuse its arena
until lucky or attribute the failure solely to value drift.

## Hypothesis and fixed control

Isolate whether distilling **the network's own fresh search policy** improves its
low-budget move choices when learned trunk/native-reference value are fixed.
Observed weak self-policy outcomes and native engine reference probabilities have
related STM W/D/L semantics but different conditional policies. Lower loss on weak
self outcomes is not automatically a better best-play value. This limited policy
iteration freezes value; it does not demonstrate a self-trained value function or
complete long-term AlphaZero learning. No Stockfish actor/label/query during play
or updates; archived teacher validation is a stopping diagnostic only.

Control: UFUK validation-selected step9250 SHA
`18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae`.
Explicit weights-only new-task warm start and optimizer reset. Only16914`pair_`
parameters trainable; every other trunk/value parameter must remain bitwise
unchanged throughout. Store selected trainable names in complete checkpoints.

## Learning, inputs, resume and budgets

- CPU FP32/one learner thread, four **spawned** one-thread CPU actors, no fork.
 Existing4CPU/16GiB allocation, no new paid resources. Immutable actor snapshot
 and drained actors before updates. Parent and child CPU measured separately.
- Seed20261006,3generations×32games,64FullGumbel simulations/max16/Gumbel1,
 stochastic first exploration plies as existing `SelfPlayConfig`,max240totalplies.
 Start each game at one of48fixed teacher **train** opening histories, cycling
 game-index mod48; ordinary initial-position exploration is no longer assumed.
 BookSHA`019c5b4a49ac287ca03effb65b33ed97daa2e702ddd4a6972491969e27912b24`.
 Embed legal opening histories/source SHA in run config; reject book/config drift
 on resume. Teacher validation/arena families never supply starting histories.
- Batch64,256updates/generation,AdamWLR5e-5,wd1e-4/clip5,rolling replay3,
 game-balanced sampling. Policy labels are current network search targets;
 terminal WDL retained and unknown caps remain masked, but frozen value receives
 no gradient. Record full model/optimizer/RNG/cursor/input-checksum states.
- Fresh process ends atgeneration1 under1200s wall, then full-checkpoint resume to3
 under2400s. No extension of this fixed phase. Stop on malformed/nonfinite/resource/
 protocol failure, retain partial shards and last complete immutable checkpoint.

Aftergeneration1, stop if known terminal replay rows<25% or development teacher
policyCE>2.78209935 (control2.63209935+0.15), or any frozen parameter/value logits
changed. Repeat retention/integrity gate aftergeneration3. No intermediate arena
or final selection based on game scores. If phase fails, retain control as reference.

## New frozen strength suite

24families/48colour-paired **16-ply** roots frozen before training, SHA
`85564bdfe43a7a2e4ed37f938cc3223038a8c76147e1bd84d6c478d1adfab06e`.
Source`docs/research/UFUK-selfplay-opening-splits-20261003.json`. Extends parent
CC0/stress roots by4seeded uniform legal moves; rejects terminal/already-recorded
roots, no engine/model-quality filtering. Whole components exclude all teacher
train/validation but reuse UFUK development arena families. These are fresh roots,
not fresh independent families/public Elo; synthetic suffix scope is explicit.
Post-generation audit all replay root/current FENs and original opening-component
membership for contamination. Any held-out root overlap disqualifies strength.

Final generation3 vs frozen control; both vsStockfish19/Threads1/Hash16MiB/32nodes.
Neural16simulations/Gumbel0/one thread,seed20261006,max240plies. Each48legal games,
same roots and both colours. Primary policy self-learning strength requires
score>0.60,paired-family bootstrap lower>0.50,caps<=10%,plus retention/integrity.
Report conservative bounds, all losses, unknowns and actual wall/NN throughput.
Automatic claim_draw protocol retained; sim/node budgets unequal, no equal-compute
or high-strength claim. Failure stays failed; no reruns/threshold changes/promotion.

Log material results Git+Space, source/hash/commands/tests and artifact inventory.
Release transport is still blocked: preserved local files and published text are
not a completed remote binary backup. Record measured failures without success washing.
