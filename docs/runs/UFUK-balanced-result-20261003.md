# UFUK broader balanced teacher learning: development strength gate passed

A fixed end-to-end model, selected **only** by development validation, beat the
frozen AYNA bootstrap20/26/2 on24fresh-root families/48colour-paired games:
score0.6875, pair bootstrap95%[0.604167,0.770833], conservative
Hoeffding[0.410279,0.964721], zero caps. Registered development gate passed.
This is teacher-assisted bootstrap improvement, **not demonstrated self-learning**,
Stockfish/AlphaZero-level strength, a general Elo, or champion promotion.
The head-only control failed its learning gate and never entered the arena.

Protocol: [UFUK-balanced-preregistration-20261003.md](UFUK-balanced-preregistration-20261003.md).
The previous AYNA self-learning strength failure remains failed, with raw evidence.

## Corrected collection and independent integrity audit

- Existing allocation:4CPU quota,16GiB, noGPU/CUDA, no purchased resources.
-1024trajectories/80511reference rows,48train/16development-validation families,
 16trajectories per family, both actor sources and STM colours, seed20261005.
-Three spawned workers, one Torch thread and one Stockfish19 thread each;
 native soft WDL and top4CPsoftmax policy at32768requested nodes/Hash16MiB.
 Teacher20%soft-policy exploration before ply40; frozen neural raw-policy actor.
-Actual80511queries/2,620,158,864nodes; collection wall1111.5780s,
 childCPU3737.5625s. Requested budgets are distinct from actual node counts.
-Data source`71f03d04cd00af91bd4bd4ba47b76a6f569597f9`;
 manifestSHA`b34a3870b494b4cc0fe8a1df87960fe5e2406a5c9a2ed341d1b93fed0e0e094f`.
 Stockfish SHA`0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19`.

|Source|White STM rows|Black STM rows|
|---|---:|---:|
|Engine/engine|30797|30589|
|Neural/engine|9690|9435|

Neural actor colour256white/256black games; both sources512games each.
979distinct full histories;53829train/17771validation distinct current-position
keys, not80511independent positions. Raw train60375/validation20136rows;
zero train/validation current-FEN overlap and zero frozen arena-root overlap.
Seeded validation cap selects20000rows; train cap80000does not truncate60375.
All1024file hashes,80511row histories/FEN/legal action sets/probability mass/
coherent completed reference depth, all final moves/endings independently checked.
864true terminal games/160unknown capped games; capped observed results remain null.
Reference targets remain native engine probabilities, separate from terminal replay.

Metadata's legacy `actor_nodes:4096` describes the fallback on **unlabelled** plies.
Balanced collection labels every ply, so that fallback was unused; engine actors
reuse the32768-node reference query. Every recorded reference budget confirms this.
Thus comparisons to the old tiny dataset change size, balance, trajectories and
actor budgets together; no claim that colour balance alone caused the strength gain.

## Fixed learning controls and complete checkpoints

Both arms warm-start the same frozen AYNA step2200 SHA
`24679cbeacc2ad0a65e6e80ada5867bc833d60618a799270228ccf5c6800ade3`,
with **new optimizer state**, seed20261005, CPU FP32/one thread, batch64,
10000updates maximum,250update validation interval/1500patience,1800s wall cap.
Each completed10000updates/640000sampled rows; selection uses total validation CE.
72497total parameters; head arm trains16914`pair_`parameters atLR2e-4,
end-to-end trains all parameters atLR1e-4. Architecture/encoder/actions unchanged.

|Metric on same20000development rows|Frozen baseline|Head best9000|All best9250|
|---|---:|---:|---:|
|Policy CE|2.873652|2.791447|2.632099|
|Soft-WDL CE|1.207413|1.207413|0.654349|
|Reference Q MAE|0.583882|0.583882|0.336883|
|Top16 reference-move coverage|0.80910|0.83240|0.90035|
|Arena learning qualification|—|FAILED|PASSED|

Head policy delta0.082206<0.10 required; value unchanged as expected. No threshold
relaxation, no head arena. All deltas policy0.241553/value0.553064 satisfy the fixed
0.10/0.10 rule. Selected weights:

-Head SHA`37ee86842e183b304f98b124288832727d0784e01351c5311000e8ae59e52bc2`.
-All SHA`18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae`.

Head source`47d0535dd634f049b7648a7106f3553e8a861a65`, wall584.7644s,
 CPU587.1009s/peakRSS5332528KiB. All source`22e1b4a04bb5dbdf889db05d39f05d73d05f764a`,
 wall690.2458s/CPU692.1894s/peakRSS5332512KiB. Data preparation and full validation
 are included in session wall. Brief test/audit processes overlapped under the
 allocation; these timings are resource accounting, not isolated speed comparisons.
 All41checkpoints per arm (including step0) retain optimizer/RNG/input hashes/
 cursor/evaluation/best-selection state. This run did not itself pause mid-training;
 previous real pause/resume and separate-process regression remain distinct evidence.
 Resume must use recorded trainer/data hashes; later source edits never rewrite
 old provenance. Copying selected weights alone is not training resume.

## Matched strength and throughput

24component families/48paired colours,12-ply frozen roots SHA
`aec0ffe243011f3d4299b798b1608008b231e738fd35292a880e31b31086904d`,
seed20261005,16neural simulations/Gumbel0/one thread, max240plies.
Some families were used in older development arenas; these roots were never played
in those192games. Short CC0 book entries use explicitly marked uniform legal
suffixes. This is a fresh-root stress suite, not an independent public rating.

|Candidate/opponent|W/D/L|Score|Pair bootstrap95%|Wall s|Actual NN evaluations|
|---|---:|---:|---|---:|---:|
|All / frozen bootstrap|20/26/2|0.687500|[0.604167,0.770833]|154.8313|79409|
|All / Stockfish32|1/13/34|0.156250|[0.093750,0.218750]|43.9275|20741|
|Frozen / Stockfish32|2/7/39|0.114583|[0.052083,0.187500]|44.5174|21492|

Stockfish19/Threads1/Hash16MiB/32requested nodes, same executable hash above.
Paired SF score delta0.041667, bootstrap[-0.062500,0.135417], includes zero.
Do not claim a proven SF gain; decisive losses dominate. No general Elo.
All144games/11539plies independently replayed legal; scores/roots/colours/endings
verified; no capped games. Arena source`a62330b7e0a11127f02fc4dc7e67fabc1b632fd5`.
512.8744/472.1642/482.7772NN positions/s are actual complete-arena throughput,
not unique learning rows or backend-only inference speed. Fixed sim/node budgets
have different costs; actual SF visits are not recorded by this older arena version.
Automatic `claim_draw=True` includes claim-by-next-move; this declared protocol
is a simplified claim rule. All uncertainty is conditional on these development
families; conservative intervals are wide despite passing the bootstrap gate.

## Commands and portability validation

```bash
python -m harbichess.training.oracle_data artifacts/ufuk-balanced-data-20261003 \
  --openings docs/research/AYNA-opening-splits-20261003.json \
  --weights artifacts/ayna-pairwise-train-20261003/checkpoints/step-002200/model.safetensors \
  --stockfish /path/to/stockfish --games-per-family 16 --workers 3 \
  --seed 20261005 --balanced --wall-seconds 3600
python -m harbichess.training.oracle_train artifacts/ufuk-head-train-20261003 \
  --dataset artifacts/ufuk-balanced-data-20261003 \
  --weights artifacts/ayna-pairwise-train-20261003/checkpoints/step-002200/model.safetensors \
  --qualification-reference artifacts/ayna-pairwise-train-20261003/checkpoints/step-002200/model.safetensors \
  --trainable-prefix pair_ --qualification-kind policy-with-frozen-value \
  --learning-rate 0.0002 --max-steps 10000 --interval 250 --patience 1500 \
  --seed 20261005 --max-train-rows 80000 --max-validation-rows 20000 --wall-seconds 1800
```

All arm: same dataset/weights/reference/caps/seed/budgets, no trainable-prefix or
special qualification-kind, LR0.0001, separate `artifacts/ufuk-all-train-20261003`.
Exact arena commands and stdout are recoverable in the evidence bundle.
New immutable destinations are required for reproduction.

486tests passed/0skipped (34.24s). Changed Python files pass Ruff. Real MLX CPU
pairwise forward/masked/soft-loss/gradient/update/optimizer-snapshot tests pass.
Torch-free MLX version1 loader/exporter converts OIHW/OHWI explicitly; a fresh
interpreter actively refuses Torch imports and verifies round-trip weights and
strict finite/schema/layout boundaries. See [PORT-runtime.md](../PORT-runtime.md).
Metal/CUDA/BF16 device tests unavailable; no optimizer equivalence across frameworks.
The new engine-dataset CLI remains Torch; no MLX-native CLI claim.

## Preservation and next decision

[UFUK-balanced-evidence-20261003.json](UFUK-balanced-evidence-20261003.json)
contains18exact UTF-8 metadata/manifest/audit/train/arena files, byte counts and
SHA-256. It restores measured text, not large binary models/datasets.
Full checkpoints/replay/reference gzip files remain local. Genuine small ASCII
Release uploads failed even over HTTP2; remote binary backup remains incomplete.
Old checkpoints, failed datasets/protocol logs, rejected cache and AYNA failures
are preserved. Publishing a source report is neither backup completion nor promotion.

The successful broader teacher bootstrap supports a **new preregistered** self-
learning test. Earlier all-parameter self-learning worsened reference Q and did
not improve strength. Next isolate policy improvement from own fresh search while
freezing trunk/value, retaining the stronger calibrated model as control. This
would test one limited mechanism, not prove complete autonomous long-term learning.
