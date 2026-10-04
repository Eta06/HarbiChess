# UFUK: bounded tactical leaf result

Registered quality, cost and checked-fallback gates **FAILED**. No arena,
default search change, teacher call, optimizer update or model promotion.
Main strong/fast/reliable self-learning and Stockfish/AlphaZero-level goal remains
unmet. [Prior protocol](UFUK-tactical-leaf-preregistration-20261004.md).

Clean pinned source`023012f7a097c4526b6c32c5dfa296bceb0e039e`, original18f weights unchanged.
128frozen native development positions/16families/3433all-legal independent
Stockfish32768references reused from the prior diagnostic; repeated controls
are not fresh independent strength evidence. All three arms retain neural policy
priors and16sim/max4/G0/value_scale0.1/maxvisit_init50, CPU1/eagerB1/seed21.
Controls: unchanged neural value and simple static material. Candidate: same
material with ordinary negamax/alpha-beta capture/promotion quiescence,
all legal check evasions, depth4/hard64visited nodes per leaf, exact terminal
values/history-sensitive rules, no checked stand pat. No GPL search implementation
or Stockfish tuned constants were copied. Classical material is not learned.

| Contrast | Endpoint | Family mean | Adjusted98.75% interval | Gate |
| --- | --- | --- | --- | --- |
|quiescent−neural|selected_expected_score|0.007523438|[-0.017738281;0.033570312]|FAILED|
|quiescent−neural|soft_expected_score|0.004475496|[-0.013903557;0.023925679]|FAILED|
|quiescent−static|selected_expected_score|0.009714844|[-0.006480469;0.035187500]|FAILED|
|quiescent−static|soft_expected_score|0.005867532|[-0.007055601;0.020898944]|FAILED|

All four required mean>=0.01/Bonferroni lower>0 conditions failed. Nominal95%
intervals,16whole-family effects/eight positions each and conservative adjusted
Hoeffding bounds are preserved. Conditional bootstrap does not establish general
Elo or account for every previous exploratory model choice.

Actual **6470neural evaluations/17.773227s**,
below7500NNcalls/300s fixed caps. Quiescent calls2151, visited16086nodes,
12node-bound and2570depth-bound events,118calls with checked static fallback.
Checkedfallback fraction**0.054858>0.05**;
these recorded approximations are not pass moves or proven minimax bounds.
Median paired search wall ratio**1.685755>1.5**;
three-arm order rotated by position, no concurrent training. No inference-only
or GPU-utilization proxy; total search cost includes material/qsearch/history work.

Independent audit reconstructed every legal position/action distribution and
all four adjusted intervals, recomputed selected/soft reference endpoints,
verified per-leaf hard node budgets/counters and exact historical neural-control
selected moves/visits/policies. Four meaningful tactical tests passed0.07s;
fullsuite527passed/zero skips46.19s. Enpassant, promotion, capture/recapture,
terminal/draw/rule50/STM signs, legal check evasions and unchanged board/history
covered. Initial lint SIM113 warning corrected before final fullsuite/source
freeze; final lint and fullsuite logs are preserved. AppleMetal/CUDA unavailable; the
tactical mechanism is framework-neutral and did not modify existing MLX math.

Existing4CPU/16GiB/noGPU allocation, no paid resources. Old experiments,
models/replays and historical failures preserved. Exact UTF8 evidence is in Git;
no claim of new remote binary backup (Release400BadContentLength remains).
The small frozen-policy attention training independently failed its native gate;
this tactical test cannot retroactively change that result. Next prospective
research is a cheaper learned sparse/incremental value representation and
appropriate search controls, not an assumption that copying NNUE yields strength.
