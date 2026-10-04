# UFUK: joint trunk/policy/value adaptation after frozen-branch failures

Prospective protocol before new updates. Previous context, sparse value and
all-legal matched-wall experiments stay FAILED. This is a distinct hypothesis,
not an extension or relabelling of stopped runs. Main goal remains unmet.

Hypothesis: freezing the old16channel shared representation constrained the
policy-only context experiment. Jointly training every parameter used by policy
and value may permit useful context features. Compare original18f all-used-parameter
control to retained identity-context initial transfer e146fd, not the failed
trained context checkpoint. Initial Torch policy/value functions must be exact;
actual MLXCPU tolerance2e-5. Context mathematics is standard attention, not novel.
No new model schema or weights conversion required; current Linux and MLX support
remains. Only historical21parameter unused material auxiliary stays frozen.
Both branches train value, shared trunk and pairwise head; context also trains
its6496parameters. Registered models72497/78993, active72476/78972.

Original18f SHA18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae;
identity-context SHAe146fd4e1cc103ef42bc0fb3e7a0a7fcd3d33a2431179b9d0840757f52ec28d5.
This starts fresh optimizers. It is not resuming old failed policy-only training,
not claiming a weights-only transfer is full training resume. Preserve all old
models/replays/checkpoints/results. Within-run full native model/optimizer/RNG/
input/sampling cursor resume must be actually exercised at1000 in a fresh process;
source/config/runtime/input hashes strict. Frozen auxiliary checked every eval
and every final publication, nonfinite/mismatch refuses unsafe continuation.

Same native80511dataset/48train16validation whole families/caps40000/20000,
seed20261024, CPU1FP32/AdamW1e-4/wd1e-4/clip5/B64/max30000/eval1000/patience7500,
3600whole-wall seconds PER arm including repeated input setup and1000pause/fullresume.
Unchanged original soft teacher targets/losses/sampler; no new teacher calls.
Same sampled rows at common update counts. Select minimum native totalCE among
valid checkpoints, not a lucky checkpoint selected by games. Report both CE and
QMAE throughout. Stop budget/patience/nonfinite/frozen/runtime/source/input change;
no result-dependent extension. Old validation families are reused development
data; qualifying this panel is not independent model strength or self-learning.

Register TWO distinct model-improvement hypotheses versus their common initial
function: each selected model qualifies for games only if policyCE gain>=0.03,
valueCE gain>=0.02 and QMAE<=initial+0.01. Require all three within an arm.
Also compare context to joint control: architectural representation gain requires
policyCE>=0.03 better and valueCE/QMAE<=control+0.01; report FAILED if absent,
even if either separately beats initial. A joint control improvement cannot be
called an attention improvement. Legacy core default thresholds are not these gates.

Fixed18full-history inference probes from width stage/CPU1/eagerlegalmaskedB1,
10warmup/200alternating rounds/no concurrent training. Full policy+value task;
median model/original latency<=1.5 for each otherwise no games. Setup/encoding/
search excluded from this latency only, no whole-game speed inference.

Freeze NEW24uniformlegal12ply stage stress families seed24 before updates, all
existing source/history/first4prefix/native/prior-arena exclusions, no engine or
model quality filtering. Include unused sparse/context books. Not natural-opening
Elo or exhaustive inherited-pretraining holdout. If either qualifies, selected
versus original and selected versus SF512,48colour-paired games per arm, same
fresh24family book/16sim/max4/G0/value_scale0.1/maxvisit_init50/max400plies,
CPU1eagerB1/seed24/900whole-wall per48games. Shared original-versus-SF51248games
reference runs once if any qualify. Both hypotheses adjust candidate-vs-original
family bootstrap intervals to97.5% (10000replicates/seed24), conservative Hoeffding
also recorded. Strength gate score>0.60/adjusted lower>0.50/caps<=0.10. SF19Threads1/
Hash16MiB/requested512actualnodes recorded; paired SF delta97.5% uncertainty.
Same simulation counts but different actual latency, engine nodes incomparable
to simulations: report total game/move wall, median/p95, calls, requested+actual
nodes, colour/opening pairs and all legal traces. No general Elo/Stockfish-level claim.

No promotion or new training merely because native loss improves. A qualified
bounded game pass would need a fresh closed-loop experiment with same-search
initial/final independent pairs before any self-learning claim. Existing4CPUquota/
16GiB/noGPU only, no new paid resources. Laya/LLM datasets remain later research.
