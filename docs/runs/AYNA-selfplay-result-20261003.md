# AYNA self-learning strength failed; CPU speed controls

The registered final-vs-bootstrap strength gate **failed**: 5 wins, 31 draws,
4 losses, score0.5125 on20 fresh colour-paired opening families/40 games,
bootstrap95%[0.4375,0.5875], conservative Hoeffding[0.208819,0.816181].
Required score>0.60 and bootstrap lower>0.50 were not met. No capped games,
no rerun, no generation selected by arena results, no champion promotion.
The previously qualified pairwise bootstrap stays the development reference.

## Actual learning and restart

Preregistered protocol: [AYNA-selfplay-preregistration-20261003.md](AYNA-selfplay-preregistration-20261003.md).
Initial weights-only new-task warm start SHA
`24679cbeacc2ad0a65e6e80ada5867bc833d60618a799270228ccf5c6800ade3`,
optimizer explicitly reset. This is teacher bootstrapping provenance, not pure
zero-knowledge AlphaZero. Subsequent generations used full optimizer/RNG/cursor/
replay checkpoints. Fresh-process resume after generation1 completed2 and3.
Final model SHA`aca25f6104df3c9d9fcf743df3e56bc7024f2ef166e0170d871e7a183e59d1a6`.
No teacher actors or teacher labels were used in these self-play updates.

CPU FP32, one Torch thread/four shared thread actors, seed20261004,
FullGumbel64/max16/Gumbel1,16games/generation, max240plies, batch64,
64AdamWupdates/generation/LR2e-5, rolling replay3. All72497parameters trainable.

|Generation|Games|Replay positions|Observed terminal games/rows|Self-play wall s|NN evaluations|Training wall s|
|---|---:|---:|---:|---:|---:|---:|
|1|16|2765|10/1325|380.2386|178273|2.2039|
|2|16|3218|9/1538|471.2531|207879|2.3066|
|3|16|2661|10/1221|392.2648|171378|2.4150|

Totals:48games,8644positions,29observed terminal games/4084known rows,
192updates/12288sampled rows,557530NN evaluations. Capped outcomes stay unknown
and value-masked. Unique replay positions differ from search evaluation count.
The resumed process wall946.6856s/parentCPU946.4505s/peakRSS2496468KiB;
its source`87e7f2082b820fc7e3fd23fc143dfbc8131a8ee7`. This excludes the
first process; generation1 times survive in its immutable checkpoint and log.
Do not add resumed resource totals to an invented first-process total.

Whole-game-held-out rolling self-outcome WDL CE1.803853→1.710644→1.680237→1.648457.
These are outcomes of weak players and evolving panels, not optimal reference value.
Teacher development retention passed: CE0.647972→0.672103<0.797972 ceiling.
However reference QMAE0.332212→0.388216 and policy CE2.966929→3.114917 worsened.
Thus lower self-outcome loss is **not** proven playing-strength improvement.
Self-play initial-history audit found no overlap with the20 held-out components.

## Fresh matched strength controls

Seed20261004,20families×both colours,16neural simulations/Gumbel0,
Stockfish19/one thread/Hash16MiB/32requested nodes, max240plies, same roots.
Suite SHA`174c94e72dcfec8fc946bc0dd34d751b00f8cbce12108445153f078a2271667d`.

|Candidate/opponent|W/D/L|Score|Pair bootstrap95%|Wall s|
|---|---:|---:|---|---:|
|Final / bootstrap|5/31/4|0.5125|[0.4375,0.5875]|129.4414|
|Final / Stockfish32|0/16/24|0.2000|[0.1500,0.2500]|37.7260|
|Bootstrap / Stockfish32|0/11/29|0.1375|[0.0750,0.2000]|37.2729|

All120games were independently replayed for legal moves, starting roots/colours,
claimed termination and scores; exact audit and paired SF delta are in the evidence.
SF paired delta0.0625, bootstrap95%[-0.0125,0.1375], includes zero; it
is secondary and cannot override the failed primary.
No Stockfish wins, no SF/AlphaZero-level strength, no general Elo.
Final arms source`cd7e5a9628a46fc9a301905e4265d40da409582d`; bootstrap-SF
source`bb506bad8f2cd5644b86453481c72a8b9a262537` (documentation differs).

The arena automatically accepts `claim_draw=True`, including claim-by-next-move.
This is a declared simplified draw-claim protocol, not optional player claim action.
Neural simulations and SF nodes have unequal wall costs; the arena records move
timing but not actual SF node visits. Fixed-budget games must not imply equal compute.
Bootstrap intervals are conditional on these fixed families; conservative bounds
remain wide. This is development evidence, not a public rating tournament.

## Measured speed: successful and rejected mechanisms

All controls preserve frozen weights24679cbe…; no strength inferred from throughput.

- Legal-only pairwise masked inference: one-thread batch1,30repeats,
  910.4156→1086.4324positions/s,1.1933×; batch8 1910.4129→2331.0261,
  batch32 2366.4236→2830.8374. Includes tensor/host/backend work, excludes
  chess encoding/search. FP32 logits/loss/grad parity uses2e-5 tolerance,
  not bitwise equality. Source48a1f0f…→9002909….
- Spawned independent CPU actors: eight actual64-ply/16-simulation capped games,
  four actors/one thread each, identical seed and8703NN evaluations,
  wall14.8111→8.3864s including model loading/worker startup, unique replay
  throughput34.5687→61.0510positions/s,1.7661×. ChildCPU29.7968s is counted,
  parentCPU0.6227s, largest worker319268KiB. No fake multiprocessing or GPU.
  Thread default remains compatible; process mode is explicit and rejects a
  cross-mode checkpoint resume. Tests perform real spawned play/train/full resume.
- History-keyed immutable rule fact cache: wall15.9345→16.2807s,0.9787×,
  failed preregistered1.15× gate. **Rejected and removed from default rules**.
  Experimental implementation and tests stay in explicit isolated module;
  complete histories preserve repetition/fifty-move claims. No speed gain claimed.

The speed games are all capped/unknown and not a strength test. Raw replay byte
hashes differ across scheduling arms; exact legal histories/actions/record
comparisons are recorded in final-audit.json. Full bitwise trajectory equality
is not claimed. The short cache control overlapped low-cost retention inspection
briefly; this neither establishes a winner nor justifies enabling a failed cache.
The registered self-learning run kept thread actors; do not credit its games with
later process throughput. Metal and CUDA device measurements remain unavailable.
Real Linux MLX CPU/Torch parity runs; no Apple device or optimizer equality claim.

## Next decision and durable evidence

The old3601-row teacher panel structurally confounded actor source with absolute
STM colour:2756engine/engine rows all white;845neural/engine rows all black.
This is an observed sampling flaw, **not a proven causal explanation** of failed
self-learning. UFUK preregisters both-source/both-colour broader data and head-only
vs all-parameter controls, retaining all AYNA failures and old checkpoints.

[AYNA-selfplay-evidence-20261003.json](AYNA-selfplay-evidence-20261003.json)
contains verbatim UTF-8 arena/session/retention/audit/benchmark files with byte
counts and SHA-256, recoverable without recomputing results. Full replay shards,
optimizer files and model weights stay local. Existing GitHub Release upload
transport rejects even genuine small ASCII/HTTP2 inventory; **remote binary
backup is incomplete**. No asset readback, no remote model/data preservation claim.
Use original source checkout for changed trainer/data code checkpoint resumes;
weights-only conversion is never full training resume.

## Reproduction commands (recorded configuration)

Run from the original source checkout with existing `.venv`; preserve relative
replay paths when restoring an archive. These are equivalent commands, not a
claim that changed code reproduces historical floating-point trajectories.

```bash
python -m harbichess.training.torch_loop artifacts/ayna-selfplay-20261003 \
  --weights artifacts/ayna-pairwise-train-20261003/checkpoints/step-002200/model.safetensors \
  --generations 1 --wall-seconds 900 --seed 20261004 --games 16 \
  --simulations 64 --gumbel-scale 1 --max-plies 240 --workers 4 --threads 1 \
  --batch-size 64 --steps 64 --learning-rate 0.00002 --replay-generations 3
python -m harbichess.training.torch_loop artifacts/ayna-selfplay-20261003 \
  --resume artifacts/ayna-selfplay-20261003/checkpoints/generation-000001 \
  --generations 3 --wall-seconds 1800 --seed 20261004 --games 16 \
  --simulations 64 --gumbel-scale 1 --max-plies 240 --workers 4 --threads 1 \
  --batch-size 64 --steps 64 --learning-rate 0.00002 --replay-generations 3
python -m harbichess.evaluation.portable_arena \
  artifacts/ayna-selfplay-20261003/checkpoints/generation-000003/model.safetensors \
  artifacts/ayna-pairwise-train-20261003/checkpoints/step-002200/model.safetensors \
  --openings docs/research/AYNA-selfplay-opening-splits-20261003.json \
  --opening-pairs 20 --seed 20261004 --simulations 16 --max-plies 240 \
  --output artifacts/ayna-selfplay-arena-20261003/vs-bootstrap.json
```

Existing destinations are immutable; do not rerun over recorded artifacts.
Actor benchmark module `harbichess.benchmarks.actors` takes frozen weights,
new output directory, `--mode thread|process`; all exact result configs/sources
and weights hashes are in the evidence bundle.
