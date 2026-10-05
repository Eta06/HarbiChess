NEW SCRATCH CODE ONLY. No actual data prep/fit/search qualification/games.
268 sparse action features: piece6/from32/to32/piece-to192/phase4/castle2,
mover-relative ranks + file folding. No gives_check, NN or oracle input. Human
prior18 theta0 evaluator unchanged. CE only own-search selected quiet move,
NOT exploration action; exclude every explored row including random equality.
Canonical known FINAL16384 full-history trajectory dedup/protected game/hash
split identical to original learner; UNKNOWN/tails excluded. Minimum16distinct
quiet trajectories/1024quiet rows/nonempty splits, derived updates>=8. Game-
uniform then row-uniform sampling, stable ALLlegalquiet softmax. Fixed Adam.01,
b256/4Ntrain slots/max1024, clipnorm5, L2.001 meanweights². No target teacher.
Endogenous weak-expert imitation (ExIt), no optimal-action/novelty/strength claim.

ordered_search.search_type(SHA_VERIFIED_ORIGINAL_BudgetSearch) overrides ONLY
ordered(). Capture/promotion original slots unchanged; quiet indices reorder.
ALLroot coverage, root score sorting, evaluator, q2/depth8/node512 unchanged.
Root loop itself does not consult ordered(): improvement must generalize into
interior quiet ordering, not change root policy directly. Theta0 exact original
order/returned Result fields (there is no exposed PV packet). Zero model skips
feature work. Both baseline and learned run SAME new module, same θ0 evaluator.
Learned feature overhead may worsen latency; must measure, not assume faster.

ROOT freezes source/helper/input/evaluator/search hashes and original clocks.
Future bounded native proof: qualify.py --protocol FILE --protocol-sha256 SHA
--config ACTORCONFIG --journal FINAL16384 --source-repo CLEAN6FCC --output NEW
--seed SAMESEED --cpu-core 1(or2) --first OBSERVED --deadline FIRST_PLUS600
whole8/pause4/freshresume8 +six freshloads0/4/8. Native JSON contains all268weights,
Adam m/v/step/global and sampler RNG/data/source/protocol hashes. Distinct schema
own-quiet-ordering-native-v1; never reuse value/actor native. Main fresh final
fit via train.py, no --resume or --stop-at, new registered deadline. Outputs
candidate.json only derivedfinal update; all checks/audits consume own deadline.
Qualify imports unchanged actual classical ownership/runtime/publish closure,
SHA-bound in protocol, loaded only after dependency validation. Parent ROOT must
also freeze qualifier/search adapter bytes (included ordering closure). Runtime
physical/OOM/16MiB and workspace256MiB guard retained; CPU affinity explicit.

Later known160 development: learnedorder vs SAMEθ0order, E8, SF512; same frozen
humanprior for both order arms, E8 own evaluator only reference. Same captures/
search settings/roots/rules, counters+latency required. Improved own CE/depth is
NOT strength. Formal independent confirmation only if prospectively selected
under ONE-new-campaign accounting, with unchanged strong prior/E8/SF gates.
Five tests PASS0.19s: exactzero order/32-node tiny fake-value Result identity,
capture slots, reflection, gradient and synthetic 8/4/fresh8 native+RNG. This
is not real evaluator/search or actual16384 SGD qualification. Parent supplies
owned phase controller before execution. No automatic train/game launch.
