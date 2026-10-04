# UFUK: CPU tactical mechanisms after repeated learning failures

Primary source read while the separately pinned policy-context native experiment
was running; no new tactical experiment, training or strength result is claimed.
The ordinary attention branch remains its preregistered hypothesis.

Stockfish19 source tag resolves to
`edb0d9db6731067ec50ce619ff372b463bc4dd5d`.
[search.cpp](https://github.com/official-stockfish/Stockfish/blob/edb0d9db6731067ec50ce619ff372b463bc4dd5d/src/search.cpp#L1646)
is 94212bytes, SHA256
`a934524dd2f386ec38cdf95b7b2e41cecdc85c9b17e12c0668621e6ae16be28a`.
[evaluate.cpp](https://github.com/official-stockfish/Stockfish/blob/edb0d9db6731067ec50ce619ff372b463bc4dd5d/src/evaluate.cpp#L41)
is 3844bytes, SHA256
`68a1675ef7ae943026ca0e54368b88ab90dd0082d22b5504f743afb2765dffb7`.
Exact snapshots and fetch receipts are retained for the next evidence bundle.

At depth zero Stockfish searches tactical moves instead of always trusting a
static evaluation. Its qsearch treats being in check separately, permits stand
pat only outside check, searches captures and check evasions, uses static exchange
and futility pruning, and handles terminal/repetition rules and transposition
bounds. Static evaluation is side-to-move and explicitly assumes no check;
NNUE runs through accumulator stacks/caches, combines PSQT/positional terms and
scales with material and rule50. Its tuned constants and GPL implementation are
not copied into HarbiChess. Reading these sources is not a measured improvement.

HarbiChess's current CNN leaf evaluation is comparatively expensive, and its
learned value remains weak against even the bounded SF512 reference. These are
measured limitations, not proof of a single cause. Two next hypotheses are worth
separate controls: a bounded tactical search around leaves may reduce horizon
errors; a compact sparse/incremental learned evaluator may permit more useful
search per wall-clock second. A hand-crafted material/quiescence control should
also be tested so any gain is not falsely attributed to learned intelligence.

Any implementation must preserve history-sensitive repetition, rule50, en
passant, castling, promotion, STM signs and exact terminal values. A FEN-only
transposition cache cannot silently merge incompatible histories. Hard node,
depth and wall limits must expose truncation; checked-node stand pat cannot be
treated as a legal pass. Neural and classical search must be compared on total
move/game time as well as their different internal work counts. GPU utilization,
same-node counts, or faster isolated evaluation are insufficient strength claims.

After the active context result, register the smallest diagnostic with matched
controls and immutable references before implementing production behavior.
Only a useful diagnostic should advance to independent color/opening-paired
games and then a new closed-loop learning test. No Stockfish/AlphaZero-level,
novelty, teacher-surpassing or self-learning claim follows from these standard
mechanisms; publishable contribution requires a useful independent mechanism,
ablations, broader baselines and reproducible strength/compute evidence.
