# UFUK: prospective real-checkpoint online CLI and native-resume preflight

Before any real e8 online update. Infrastructure hypothesis: the new one-ply
learner can execute actual legal rollouts and both policy/WDL updates from our
real model, preserve current/base/EMA/Adam/RNG/complete actor history and sample
chain, and reproduce the next trajectory/update after a real process exit.
This does not hypothesize a strength gain from five updates.

Code source pinned ca11c9eccc2133efb2044dead2c2084b3efb709d; exact components
and74targeted/zero-skip verification already published. Do not alter code,
coefficients, seed, roots, tolerance or budget after observing this preflight.
Execute only after current capacity training, replay audits, posttraining checks
and its conditional strength controller have terminal receipts and no owned
training/arena/benchmark process remains. Existing capacity failures are retained.

First run the complete updated regression suite from this clean source, with
CPU1/OMP1/OpenBLAS1, under180WHOLEseconds including Python startup. Require zero
failures and zero skips; the historical587-suite is not substituted. Preserve
stdout/stderr/command/whole receipt. Stop the next preflight if this check fails.

Then use fixed warm weights e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03,
`artifacts/ufuk-broad-v4-train-seed-20261026-candidate-20261004/checkpoints/step-030000/model.safetensors`.
This choice is independent of capacity results; no new seed/checkpoint selection.
Warm-start is weights-only, Adam reset, native resume thereafter. Use the complete
4096TRAIN-source-root split from existing standard/full-history book
`docs/research/UFUK-broad-history-opening-splits-20261004.json`, SHA256
f2f3038728cce4c56987c9068f900d30ed99f996f99e03a38c41bbbc1f3c9958.
Keep1024validation roots out of the actor pool; validate original prefixes/FENs
and source IDs. No source replacement, root quality filtering or new teacher query.

[Fixed configuration](UFUK-online-real-preflight-config-20261004.json):
seed20261104/four actors/four fresh transitions per update/cap8NEWplies/declared
claim_draw=True/fixed temperature1/CPU1FP32deterministic, AdamWlr1e-4/wd1e-4/
betas(.9,.999)/eps1e-8/clip5, policy-anchor.03/value-target.2/value-anchor.02,
EMAdecay.9/importance cap1. These are preflight-only constants, not chosen
production hyperparameters or a paper replication.

Maximum5updates; checkpoint every2updates, plusinitial0 andfinal5. First actual
CLI process stops at2, exits normally and saves full native state. A fresh CLI
process resumes2 and stops at5. Both pass the **same original absolute deadline**;
no budget reset on relaunch. Total900WHOLEseconds includes Python/framework
imports, all4096-root parsing/validation, input hashing, both invocations,
checkpointing, journals, every following audit and actual MLXCPU parity.
Supervisor time/resource guards apply to owned groups; failure/incomplete has
no extension, substituted source or extra updates. Memory current>15GiB/free
disk<8GiB stop. No concurrent production training or timing benchmark.

Require both CLI receipts completed, reasons respectively registered process
boundary/maximum updates, cursors2/5, and total20unique fresh transitions.
Five audit-only updates are NOT counted as production learning or improvement.
Load EVERY complete checkpoint0/2/4/5 and verify model/base/EMA/Adam/RNG/
actor/source/config/input hashes/cursor/chain. (Interval2 yields checkpoint4
in the resumed invocation.) Initial current/base/EMA tensors must equal original
e8 tensors exactly; serializer/provenance may change the weight-file SHA.
Base and unused21material tensors must remain exact across all checkpoints.
Finite actual used policy and value parameters must change after updates;
no loss-decrease criterion masquerades as game strength.

Independently restore and fully replay all20journal pre-histories/legal actions,
canonical support, declared terminal/cap conventions, actual selected pi/mu,
importance clipping, expected-score advantage and mover WDL. Verify every chain
prefix and source family; no capped draw labels. Every committed gzip byte/file
is retained and hashed. No engine query, search label or post hoc data selection.

Separately in a fresh process, load actual production checkpoint2 and replay
updates3..5 in memory under the same source/config, comparing EVERY sample,
policy/target/loss/hash, next actor state, current/base/EMA tensor, Adam state
and all RNG to the actual checkpoint5/journals bitwise. These3replayed audit
updates/12presentations are separately reported, excluded from unique progress.
No save may overwrite the production run or prior checkpoints.

Actual TorchCPU/MLXCPU full4672policy/WDL and legal-masked parity on the existing
18fixed true-history probes, SHA2561089fd0cca308c24bb040a840352d0ad18aa85a7bdcd17608133456bf08bed9f,
for checkpoint0current and checkpoint5current/EMA; atol/rtol2e-5 unchanged.
Full native states remain Torch-only: this is weight/loss portability, not
cross-framework optimizer resume. AppleMetal/CUDA/BF16 device-untested, no skips
hide absent hardware. No strength or inference-speed claim from this preflight.

Gate is conjunctive actual CLI completion + all native states + independent
journal legality/semantics + fresh-process next-trajectory/update exactness +
initial-function/base/material preservation + actual portable parity. Success
means infrastructure-qualified only. Next production self-learning must have a
separate fixed control/seed/budget/stopping plan and independent initial/final
game-strength/retention measurements. No automatic promotion or paid resources.
Store result, exact controller/audit code, commands, costs and hashes in Git;
archive real checkpoint/journal binaries when closed, retaining the known remote
Release upload limitation. The strong/fast/self-learning main goal is unmet.
