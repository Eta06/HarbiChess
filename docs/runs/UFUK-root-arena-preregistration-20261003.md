# UFUK: narrow-root diagnostic to actual matched chess games

Before any new game. The128-position mechanism test with unchanged native
bootstrap measured narrow4 versuswide16 at16sim: selected-reference expected-score
effect+0.04926172,16family bootstrap95%[0.02243359,0.07836719], registered gate passed.
These are reused native-validation families/finite references, not game strength.
Do not change the default or call it strong without this next qualification.

Freeze24 fresh24-ply roots by extending the previous20-ply UFUK selector book
four uniform sorted-legal moves perroot, seed20261014, no engine/model filter.
Reject terminal/current root already present in native data, prior arenas or
another frozen root. Keep same24 development arena families; history differs,
so this is still development selection, not a new independent Elo population.
Hash and generated book commit recorded before first game. Reference diagnostic
families are not this arena suite. No training/model update.

Three sequential arms,48 games each/24pairs: native bootstrap18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae
with max4 versus sameweights/max16; narrow versusStockfish32; wide versusStockfish32.
Bothneural16sim/Gumbel0/value_scale0.1/maxvisit_init50/CPU1/eager/batch1.
Stockfish19/SHA0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19,
Threads1/Hash16MiB/32requestednodes. Record actual engine nodes perply and pergame.
Same matched roots/colours/seed20261014, max240plies; unknown cap staysunknown,
0.5 only diagnostic score. 900s perarm, existing4CPU/16GiB; no paid resource.

Primary narrow-versus-wide score>0.60 and24pair bootstrap95% lower>0.50,
capfraction<=0.10; all48games required. Iffailed/incomplete, no promotion/default
switch or arbitrary extra seeds. Stockfish score delta paired-family bootstrap,
full setting/hash/history/endings and conservative Hoeffding reported; unequal
neural sims/SF nodes are not equal compute. No general Elo,Stockfish-level,
teacher-exceeding or self-learning claim from a search-only change. Total arena
wall/evaluations and perply wall recorded; different game lengths do not yield
equal-workload speed ratios. All played histories audited and preserved.
