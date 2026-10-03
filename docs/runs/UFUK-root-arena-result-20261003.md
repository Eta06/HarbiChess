# UFUK: narrow search passes development game-strength gate

With unchanged model weights and16 simulations, max4root candidates beatmax16
38wins/7actualdraws/2losses/1unknowncap across24new roots/48colour-paired games.
Diagnostic score0.875,24pair bootstrap95%[0.8125,0.9375], conservative Hoeffding
[0.597779,1]. Registered score>0.60/lower>0.50/caps<=0.10 gate **passed**.
This is search improvement on development families, not learned model improvement,
Stockfish/AlphaZero-level strength, general Elo or original-search novelty.

[Preregistration](UFUK-root-arena-preregistration-20261003.md), clean fixed
source79a319db917262d3b6d3d83f9c40149b267d3ee0. All144played games and11,477plies
independently replayed legally; durable start/move/end traces match completed
outputs exactly. Same original native model18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae
in both neural arms. No optimizer/model/data changes during matches.

New24-ply roots, bookSHA77a3ed4716227528a62ea52d3a77d657fe93b3ed77f449bc6fac59e41a0af69a,
uniform fourlegal suffix moves/no engine/model filter, seed20261014,24existing
development families. All80511native labels and18prior arena files excluded at
current roots. Earlier families remain reused; games cannot be pooled with prior
different roots as independent confirmatory evidence.

Allneural16sim/Gumbel0/value_scale0.1/maxvisit_init50/CPU1/eager/batch1;
max240plies/claim_draw true, unknowncap scores0.5 only diagnostic. Same matched
root/colour/seed in allthree48game arms;900s perarm,current4CPU16GiB/no paid compute.

| Arm | Wins/actual draws/losses/unknown | Score | Arena wall | NN calls |
| --- | --- | --- | --- | --- |
| Narrow vswide |38/7/2/1|0.87500|118.275778s|57,785|
| Narrow vsSF32 |7/17/24/0|0.32292|45.304528s|20,828|
| Wide vsSF32 |2/17/29/0|0.21875|36.919480s|16,996|

SF19 samebinary SHA0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19,
Threads1/Hash16MiB/32requestednodes permove. Actualnodes now recorded, including
overshoot: narrow56,937 across1252engine moves(mean45.4768,range32..276),
wide46,553 across1020moves(mean45.6402,range32..147). Node limits/simulations
are unequal compute. Narrow−wide SF score+0.1041667,paired-family bootstrap95%
[0.0,0.2083333] toucheszero: SF improvement is **not statistically established**.
Both still lose substantially to this weak fixed-node reference; no strong-engine
claim. Different game lengths prevent equal-workload speed ratios from totalwall.

The preceding128-position diagnostic predicted this root breadth/depth mechanism
with fixed reference labels. Conventional narrowing allocates more continuation
work to high-prior candidates, with coverage risk; max4 near-best-prior coverage
was0.96094 versusmax16=1.0 in that panel. No universal width/default claim at
other simulation budgets, unseen families or hardware. Old default remains
available; use explicit experimental max4 for subsequent isolated tests.

[Exact text evidence](UFUK-root-arena-evidence-20261003.json) contains every
history, move wall, actual engine node count, settings, command, root generator,
preflight failures and independent audit. A separately preregistered fresh
self-play experiment compares final versus initial weights **both with max4**,
so a later model gain cannot merely be this search change. Binary Release backup
is still incomplete; old model/replay/failures remain preserved locally.
