# UFUK: bounded tactical leaf diagnostic

Separate prospective diagnostic, production run only after the active native
policy-context training and its latency measurements finish. Its outcome cannot
change the context protocol. Use original18f model regardless of context outcome;
no new weights, teacher labels, optimizer updates or self-learning claim.

Hypothesis: current neural value misses tactical material exchanges. Bounded
capture/promotion search with check evasions at leaves may improve selected and
soft-target action quality. A static-material control distinguishes tactical
search from simply replacing the learned value. Standard negamax/alpha-beta/
quiescence and material are not novel algorithms or a trained model's improvement.
Primary Stockfish19 mechanisms were read in the separate CPU research note;
do not copy its GPL implementation or tuned constants.

Three arms share original policy priors, FullGumbel16sim/max4/G0/value_scale0.1/
maxvisit_init50, original18f weights SHA
18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae:
neural-value unchanged; static material; quiescent material. Piece units are
100/320/330/500/900, STM tanh(cp/600)*0.99. Quiescence uses no arbitrary root
move filtering: legal captures/promotions, all legal evasions when in check,
deterministic MVV/LVA then UCI order, depth4 and hard64visited nodes per leaf.
Exact terminal WDL +/-1/0, history-sensitive claim-draw rules, board push/pop
restoration. Stand pat only outside check. Hard depth/node boundaries return
explicitly recorded approximate static values when no complete checked
continuation exists; they are not legal pass moves or proven transposition bounds.
No transposition cache or hidden Stockfish fallback.

Before production, test signs, terminal/repetition/rule50, legal check evasions,
capture recapture, en passant/promotion, board/history restoration, and hard
budgets with a backend-neutral evaluator so Torch and MLX share chess semantics.
Keep all old search defaults and original code/source profiles intact.

Reuse frozen128native development positions/16families and all3433legal
Stockfish32768 references from the prior independently hashed diagnostic.
This is an exploratory development mechanism test, not new independent evidence,
game strength or general Elo. Rotate three-arm order by position, CPU1/eagerB1,
seed20261021, wall300s/NNcalls7500, no concurrent training. Record every selected
move, visit distribution, policy, NNcalls, qnodes, depth/node/checked truncations
and wall cost. No new engine queries, no gradient or candidate-driven filtering.

For quiescent minus neural and quiescent minus static, require mean>=0.01 and
Bonferroni-four-contrast bootstrap98.75% lower>0 on BOTH selected and soft-target
expected reference scores. Sixteen whole-family means/eight positions per family,
10000bootstrap samples seed21; nominal95% and conservative adjusted Hoeffding
also recorded. Require median per-position search wall quiescent/neural<=1.5,
and checked truncation fallback calls<=5% of quiescent leaf calls. Nonfinite,
illegal move, history mutation, incomplete128 or hard-budget overrun invalidates
the diagnostic; preserve durable traces and stop without extensions.

Only a complete quality-and-cost pass may advance to a separately pinned arena
implementation: new24uniformlegal12ply stress families seed21, same historical
prefix/FEN/native/prior-arena exclusions, no engine/model filtering. Freeze the
book before games. Quiescent-vs-neural and each-vs-SF512,48colorpaired games/arm,
16sim/max4,CPU1/eagerB1/max400plies/900s whole-arm. SF19Threads1Hash16MiB512requested
and actualnodes; score>0.60,24pair95%lower>0.50,caps<=0.10. Record source changes
for arena integration explicitly and recheck diagnostic semantic identity before
games; repetitions are not extra independent samples. Search-only passes cannot
establish model learning; a later fresh closed-loop training experiment would
need separate preregistration, stronger controls and no teacher-surpassing claim.
Existing4CPU/16GiB/noGPU allocation only; no new paid resource or side project.
