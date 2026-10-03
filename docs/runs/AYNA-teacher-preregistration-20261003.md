# AYNA teacher/value diagnostic preregistration — 2026-10-03

Written before probe measurements. User authorized continued research, code,
training within existing allocation, atomic commits/push and Space summaries.
No new paid resource or budget increase.

## Hypotheses

H1: existing policy assigns insufficient probability/rank to good moves.
H2: erroneous neural leaf values prevent Full Gumbel from improving choices.
H3: simply running 128 rather than 16 simulations does not reliably reduce
reference regret on this frozen diagnostic panel. These are separate hypotheses.

## Frozen panel and controls

Production starting HEAD f3f215a1fc7403aac9c12d2b9ee75459331546fb.
Use the already saved MERCEK 16-simulation arena: sixteen complete games from
eight opening families × both colors. Source file SHA-256
dcd6363d40d206a253dc51d5e5795b595d9f0e36d37531abd0b09fe91085b14d.
Choose six nonterminal positions per game at equally spaced fractions of the
post-opening history, exclude plies <8, duplicates and claimable terminal states.
At most 96 positions. Freeze full root FEN/history, game id, ply and panel hash.
This development panel is not an independent final arena or training-quality certificate.
No engine quality information is used to select positions.

Frozen initial network generation-000000 SHA-256
a155be94c96227815a2cac6f235c06622a5fb1df93a215b4ba391be500aaaa61.
Also evaluate frozen PORT generation-000003 raw policy/value (no further training)
to characterize drift. Record its original model SHA and source unchanged.

Per position:

1. Raw initial policy's argmax/legal ranks and STM WDL.
2. Full Gumbel 16 and 128, considered actions=16, Gumbel=0, unchanged scales.
3. Same learned priors/search at 16, with only leaf value replaced by Stockfish
   at 1024 nodes. This is an oracle-assisted diagnostic, not a deployable neural
   strength result or an equal-compute comparison.
4. Cold-cache Stockfish 4096 and 32768-node top moves/STM score/WDL to assess
   reference stability; all examined unique choices scored with a forced root
   move at the same 32768-node budget. Include both reference top moves and the
   frozen PORT latest raw choice in this candidate set.

SF19 binary SHA 0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19;
one thread, Hash=16 MiB, UCI_ShowWDL, no pondering. New game identity per analysis
to clear hash between queries. Save actual nodes/depth/score/mate/WDL, root move,
budget, source/model/panel SHA and timings. Scores/WDL always root side-to-move;
known terminal rules bypass inference. CP and WDL reference targets are not
observed self-play outcomes and are never relabeled as terminal replay.

## Compute, integrity and stop

Existing CPU FP32 allocation, 4 CPU quota/16 GiB, torch one compute thread.
Probe wall ceiling 900 seconds, per-UCI communication timeout 15 seconds.
JSONL append+flush after each completed position; partial runs stay incomplete
and cannot silently overwrite completed records. No concurrent training.
Stop on budget, illegal move, nonfinite output, schema mismatch or engine failure.
Only infrastructure repair before producing measurements can relaunch unchanged
arms; any changed arm/selection after data requires a new preregistration.

## Analysis and decision

Reference regret = best forced-root expected score among examined candidates
minus the arm's forced-root expected score, where expected score=(W-L)/2+0.5.
This is a finite-budget restricted reference, not true chess optimality.
Negative score differences due to varying tree budgets do not become false
“better than Stockfish” claims. Report CP separately, with mate flags.

Report top16 teacher coverage, reference agreement (4k vs32k), mean regrets by
arm, value error versus reference and versus actual outcomes separately, and
timing. Primary mechanism contrast is oracle-value16 minus neural16 regret:
average within opening family, paired bootstrap (10k, seed 20261003) plus broad
Hoeffding interval for eight bounded families. No general Elo or promotion.

Mechanism evidence requires at least 0.05 mean expected-score regret reduction
and conditional paired bootstrap lower bound >0; conservative interval remains
reported and limits inference. Otherwise the result stays inconclusive/failed.
Next training selection follows measured deficits; it can be supervised
warm start/reanalysis plus fresh self-play rather than a mandatory old architecture.
Heldout games/families for final strength are created separately before candidate
selection; these diagnostic rows cannot become its independent test.
