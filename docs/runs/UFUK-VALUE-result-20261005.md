# UFUK-VALUE — CPU own-outcome linear critic: FAIL strength

GPU access remains revoked; no GPU/SSH resources used. The strongest independently validated checkpoint remains the teacher-origin original e8 (`e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03`), not a promoted self-learning checkpoint.

Hypothesis: a simpler own-outcome critic could improve calibration and search. This is standard linear value approximation, with no novelty claim. Source is immutable `4cae08522ac940746cf9127ca8ce4a6b63a47408`, with separately SHA-bound helpers and prospective protocol in `experiments/ufuk/cpu-outcome-value-v1/`.

Two seeds 20262305/06 each reused four audited historical own-play journals (epochs 1,8,16,32; 131,072 old transitions/seed). No new self-play or teacher queries were generated. Complete own terminal outcomes label mover-perspective W/D/L; UNKNOWN games are excluded after legality/history replay. The original teacher policy/trunk remain bit-identical. An explicit zero-value rebase precedes training of only 63 parameters; an untrained rebased model is a mandatory control. No reinitialization is counted as learning.

Five tests passed, zero skipped. Exact feature/production-network parity and frozen policy checks passed. The actual whole8/pause4/fresh-resume8 training proof passed in 116.091 seconds on its original 600-second clock; all six full native checkpoints were strictly opened in fresh processes. Native v2 preserves model, Adam, global/sampler RNG, dataset split and immutable replay bindings; it is offline training resume, not actor resume. Both fixed 2,048-step fits and fresh initial/final native loads passed. Internal game-equal validation NLL was 0.73764/0.79773; this is own-policy calibration, not independent strength.

| Seed | Trained vs e8 | Trained vs zero-value control | Trained vs SF512 | e8 vs SF512 | Zero vs SF512 |
|---|---:|---:|---:|---:|---:|
|20262305|0.53125 (5W/7D/4L)|0.90625 (14W/1D/1L)|0.03125 (0W/1D/15L)|0|0.03125|
|20262306|0.56250 (8W/2D/6L)|0.87500 (12W/4D/0L)|0.09375 (0W/3D/13L)|0|0.03125|

All 160 fixed games, 13,223 legal continuation plies and 1,633,277 actual Stockfish nodes were independently replay-audited. One capped game occurred in seed05 vs e8 (6.25%, exceeding the unchanged 5% arm limit). All score/legality/provenance checks passed; strength screen FAIL for both seeds. Neither exceeded direct e8 >0.60, paired SF gain >0.10 or final SF >=0.25. Seed05 also failed the additional SF improvement over the untrained control.

The same eight known root/color families were used with S16/root-actions4/G0, 400-ply cap, Stockfish19 512 nodes/Threads1/Hash16. Paired bootstrap/Hoeffding intervals are descriptive, not virgin confirmation or generalized Elo; seeds do not make the shared families independent. Thresholds remain unchanged, no best-checkpoint selection, no model promotion. The observed improvement over a zero critic does not meet the task.

Measured limitation: these 20 invariants include material counts and metadata, but no piece placement. Next prospective method will use the existing cross-backend sparse positional value head on the same own-outcome data, with frozen policy and an untrained architecture control. This hypothesis is not yet a result. CUDA/Apple hardware remains untested. Historical failed/incomplete results and artifacts are retained; remote CPU checkpoint backup remains PENDING until upload/readback verification succeeds.

Commands and source/model/data SHA bindings, original resource clocks and complete results are in the protocol and `actual-result/` receipts. Local full natives are in `/workspace/work/harbichess/cpu-outcome-value-v1-actual/fits/`.
