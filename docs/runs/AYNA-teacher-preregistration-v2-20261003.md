# AYNA teacher diagnostic v2 — 2026-10-03, before relaunch

The original run stopped incomplete after 1/96 positions in 1.570 seconds:
"bounded engine score is not an exact candidate reference". No strength,
mechanism or calibration conclusion from that single row.
Original artifacts/source 762d19a03d0de57e9369a4730b6f05676db7838a remain unchanged.
Original failed query's raw UCI packet was not saved; this is a provenance limit.

Python-chess AnalysisResult.post updates an aggregate dict. Earlier lowerbound/
upperbound flags can survive later unbounded scores, and node-limited aspiration
can stop mid-iteration. v2 uses streaming UCI info and the **last complete
unbounded score+WDL+PV packet**, rather than treating aggregate flags as fresh.
Record completed-iteration nodes/depth and actual total nodes separately, plus
engine bestmove. If no coherent unbounded packet exists, stop incomplete.
No bounded score is reinterpreted as an exact one. Fifteen-second watchdog closes
an unresponsive engine. This is a protocol repair, not a changed strength arm.

All frozen panel/model/engine hashes, hypotheses, selection, node/search budgets,
family uncertainty, success thresholds and 900-second ceiling from the
[original preregistration](AYNA-teacher-preregistration-20261003.md) remain fixed.
Run once into a new directory; retain the original incomplete run.
The last completed iteration is a finite-depth reference, not chess optimality.
Code regression checks stale flag retention, incomplete aspiration packets,
STM mate/WDL signs and full legal history before the v2 source commit.

    .venv/bin/python -m harbichess.evaluation.teacher_probe artifacts/mercek-search-20261003/arena-16.json artifacts/port-loop-20261002/checkpoints/generation-000000/model.safetensors artifacts/port-loop-20261002/checkpoints/generation-000003/model.safetensors /workspace/work/harbichess/stockfish/stockfish-linux-x86-64-universal artifacts/ayna-teacher-20261003-v2 --wall-seconds 900
