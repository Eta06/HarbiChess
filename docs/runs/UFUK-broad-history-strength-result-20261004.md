# UFUK: broader teacher model, five-arm actual strength result

**The preregistered overall strength gate FAILED. Speed and every-game integrity
passed. The broader candidate beats its own initial/control models, but does not
establish an improvement against the bounded Stockfish512 reference.**
The strong, fast, genuinely self-learning main goal remains unmet. No promotion,
general Elo, Stockfish/AlphaZero level, teacher superiority or novelty is claimed.

[Prospective protocol](UFUK-broad-history-strength-preregistration-20261004.md),
[fixed verifier sources](UFUK-broad-history-verification-controls-20261004.json)
and [prior two-seed learning result](UFUK-broad-history-training-result-20261004.md)
are retained. Both learning seeds passed; only the preregistered primary seed
20261026 selected30000 candidate and selected2000 control played, without seed
substitution, checkpoint selection using games, extra games or threshold changes.

48 new complete-history CC0 source-game roots, both colours, 96 games per arm;
FP32CPU1, FullGumbel16sim/max4/Gumbel0, Stockfish19 full strength/Threads1/Hash16/
512 requested nodes per move, max400 total plies including each human prefix.
The previously declared claim_draw=True convention applies to both sides.
Each arm had3600 whole seconds; no concurrent benchmark or training occurred.
Neural simulations and engine nodes are unequal compute; this is a controlled
development comparison under the declared budgets, not an equal-work Elo test.

| Arm | Wins / natural draws / losses | Score | Whole seconds |
| --- | --- | --- | --- |
|candidate-vs-initial|56 / 25 / 15|0.713542|282.084|
|candidate-vs-control|49 / 30 / 17|0.666667|350.104|
|candidate-vs-stockfish512|2 / 10 / 84|0.072917|96.033|
|control-vs-stockfish512|2 / 8 / 86|0.062500|102.035|
|initial-vs-stockfish512|2 / 10 / 84|0.072917|88.023|

All480games completed; ZERO unknown ply caps. Independent python-chess replay
verified46,745legal plies including roots and31,225actually played new plies,
full prefix histories/FENs, no play beyond declared terminal convention, durable
per-move journal agreement and actual engine node counters. Thus no unknown
game was converted to a terminal draw or used as a successful outcome target.

Three primary comparisons use50,000 whole48source-game colour-pair bootstrap
resamples, seed20261027, linear empirical quantiles, Bonferroni alpha0.05/3:

| Primary contrast | Mean | Adjusted98.333333% interval | Registered decision |
| --- | --- | --- | --- |
|candidate-vs-initial|0.713542|[0.619792, 0.802083]|PASS|
|candidate-vs-control|0.666667|[0.598958, 0.734375]|PASS|
|stockfish-candidate-minus-control|0.010417|[-0.052083, 0.072917]|FAIL|

Both direct scores exceed0.60 and adjusted lower bounds exceed0.50. However
the SFcandidate-minus-control gain0.0104167 is below the registered0.10 and
its adjusted lower bound is negative. The candidate's SFscore0.0729167 equals
the initial model's score. Overall strength remains FAILED; direct model gains
are bounded positive evidence, not grounds to relabel the entire trial PASS.
Ordinary95% and conservative Hoeffding intervals are retained in exact evidence.
Family bootstrap uncertainty is conditional on these development roots; shared
chronological PGN origin, historical choices and inherited pretraining exposure
limit generalisation. No population-wide Elo is inferred.

Same fixed18full-history probes/10warmup/200rounds/three rotated model arms:
legal-maskedB1 full policy+WDL medians initial0.705665ms, control0.705039ms,
candidate0.704463ms. Candidate/initial0.998297 and candidate/control0.999183
both pass the<=1.10 gates. Setup/encoding are excluded from these warmed calls;
parent latency whole12.004s and game-audit whole8.003s include process startup.
Actual per-arm game wall, new plies/s, neural positions, move medians/p95 and
engine work are recorded separately. Different game lengths do not measure an
isolated inference-speed improvement; candidate has the same architecture.

Engine actual nodes: candidate1,225,569 across2,388moves (512..558),
control1,321,472 across2,575moves (490..558), initial1,144,285 across2,229moves
(490..561). Requested512 is not fabricated as an exact actual count: terminal
searches may finish below the request and finite search units may overshoot it.

Arena source70c612572e03f562f21469e546823244083a3b4a;
independent paired-comparison source60bbf7b60353a7b941052fae23e7f041ca37964c.
Frozen opening SHA d3b9a49e3c60cdce62a672fbfc8ae8ab2a5167ed38cc5781432a8559e0891765.
Candidate SHA e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03;
control SHA6484ccd5ba42811e41d39b52ec41b58f09cffd304442d162e99c498ec5513f6c;
initial SHA18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae.
Training/native integrity84checkpoints/four real fresh-process resume audits,
trained Torch/MLXCPU parity and current587pass/zero-skip source suite are described
in the preceding learning report; AppleMetal/CUDA/BF16device remain untested.

[Exact UTF8 evidence](UFUK-broad-history-strength-result-evidence-20261004.json)
includes every480game, complete move journals, commands/whole-clock receipts,
all fixed verifier sources, independent replay/statistical checks, measured
latency samples and Space27 history-preservation readback. Earlier failed
training/self-play/search/data-colour hypotheses remain unchanged.
All models/full native state remain in local verified archive chain; configured
GitHub Releases upload400BadContentLength still leaves remote binary backup
incomplete. Text evidence in Git does not replace model/replay backup.

Next decision: diagnose candidate policy ranking and leaf-value quality on a
prospectively frozen diagnostic panel, including measured larger-search cost,
before another self-learning or architecture intervention. Existing arena roots
are now observed development data and cannot serve as a new final holdout.
No new mechanism, training, successful self-learning or final benchmark is
claimed merely because this diagnostic is proposed.
