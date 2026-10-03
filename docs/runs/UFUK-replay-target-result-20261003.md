# UFUK: actual stored self-search target diagnostic passed

The actual stored search probabilities beat their stored raw-policy probabilities
on the frozen 96-position replay panel: mean native finite-Stockfish expected-score
effect +0.0378478058, 48-family conditional bootstrap 95% [0.02062694,0.05934194].
Registered >=0.01/lower>0 gate passed. Conservative Hoeffding remains wide
[-0.35420233,0.42989794]. This measures target quality, not played strength,
successful self-learning, new-family generalization or teacher-exceeding evidence.

[Preregistration](UFUK-replay-target-preregistration-20261003.md);
clean collector d0540e794dbabe35d25a1afc75e32897b4f40705,
SHA f1444dd62b1ed3d44a885bff1ff457bf67e7b815dede9a461ba2fb26ecf5f55a.
All six original shards and actor checkpoint checksums verified. Source 96 games,
14,885 records, original replay2/encoder1/action1/target12. Choose one white-STM
position from game f and one black-STM from f+48 for each of 48 existing actor
opening families, uniformly with seed 20261011, without quality filtering.
Panel SHA 10dc7603b884f26172c6be8b72c11745471ce2435d44a40de21a8e9b6c3b6f93.
Teacher search was not rerun. Probabilities and selected actions are the original
stored data, from changing source checkpoints across generations.

Stockfish19, one thread, Hash16MiB, fresh TT, native root-STM W/D/L, all 2,708
legal moves queried with 32,768 requested nodes; actual total 87,359,807 nodes.
Complete unbounded score packets, 15-second query watchdog. The bounded
reference is not exact minimax. Actual wall 111.515323s, parent CPU31.689308s,
peak RSS440312KiB, engine-child CPU excluded; 1800s cap, current 4CPU/16GiB only.

Independent audit reconstructed all 96 panel rows exactly from the original
source, replayed complete legal histories, checked probability normalization,
all legal/root choices, native score expectation, packet and node totals, and
family aggregation. 50 position effects positive, nine negative, 37 approximately
zero. Observed original outcomes: 33 wins,25 losses,9 draws,29 unknown. Unknown
outcomes stay masked; the engine reference is not substituted as a terminal
training label. No model or old replay changed.

The failed prior head-only online update remains failed. This positive diagnostic
justifies a separately registered value-safe policy representation comparison,
not promotion or an assertion that more training must succeed. Shared original
stem/blocks also feed value; the new branch must operate solely on policy features.

[Exact text evidence](UFUK-replay-target-evidence-20261003.json) preserves all
queries, probabilities, histories, source hashes, command and independent audit.
Binary backups remain incomplete because the authorized Release upload route
returns400; local original models, full checkpoints and replays are retained.
