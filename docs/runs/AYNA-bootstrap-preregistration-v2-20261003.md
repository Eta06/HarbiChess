# AYNA bootstrap v2: before fresh data generation

The first collection at source 4455ec4 failed with `duplicate MultiPV moves`.
Thirteen completed game files and the process log are retained under
`artifacts/ayna-bootstrap-data-20261003`. No complete dataset manifest was
published and no learner consumed those files. Raw failed UCI packets were not
captured by v1; this provenance gap limits a claim about that particular packet.

The reader grouped scores by depth, but Stockfish can publish another MultiPV
ranking at the same depth. Updating PV 1 could therefore be combined with the
previous PV 2. The repaired reader resets on every PV 1 and requires consecutive
unbounded PVs 1..K at one depth. A bounded or out-of-order packet invalidates that
cycle; only the last fully completed cycle is used. Regression covers a same-depth
swap and an incomplete bounded update. New failures save the raw UCI packets,
position/history and job provenance.

This is a protocol repair. All frozen opening groups, actors, supervision,
budgets, optimization, qualification and arena gates in the original registration
remain unchanged. Run once into `artifacts/ayna-bootstrap-data-20261003-v2`,
with the repaired source committed before launch. V1 remains failed and preserved.
