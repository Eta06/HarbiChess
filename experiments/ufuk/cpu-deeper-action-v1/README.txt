Own deeper-search selected-ACTION distillation, standard Expert Iteration.
Reviewable new producer/learner/native/student engine; no jobs executed.

Exact fixed design chosen BEFORE any optimizer update:
- Original admitted TRAIN trajectories and EXACT original1024 selector seed05/06.
  select_roots.py bytes are identical ca571105...; _position_groups AST identical
  to active value reanalysis. Original journal/config/dataset/native0/protected
  keys/clean core6fcc/helper commit/sourceSHA checks are retained in new producer.
- Frozen classical human-prior evaluator, 8192nodes/q2/depth8; no external teacher,
  Stockfish label, heldout, new quality filter, winner filter or selection sweep.
- New schema own-search-deeper-bestmove-labels-v1: actual selected UCI, all legal
  UCI candidate ordering and sparse features, full root+prefix history, state18,
  mover/rawq/nodes/evaluations/depth/root_actions/row+trajectory/source hashes.
  Bestmove hard CE ONLY; no fabricated visits/search pi or new outcome labels.
- Sparse280 = existing268 geometric/phase/castling code generalized to ALL legal
  moves, plus six capture-victim and six promotion slots. State18 is context,
  not standalone repeated bias: shared state-only logits cancel exactly in CE.
- Fixed64 Adam updates, batch256, game-uniform then row-uniform, lr.01,
  beta1.9/beta2.999/eps1e-8/clip5/L2.001 (normalized280). No VAL-selected model.
  Zero model initialization, complete weights/m/v/step/globalPython+sampler RNG
  and original input/source/protocol/deadline contract native. No NumPy/Torch RNG
  exists in this stdlib learner; it does not claim framework/native equivalence.
- Student SAME original prior evaluator, alllegal coverage512/q2/depth8.
  Main ordering signal is original previous-pass search score.
  correction=.01*(legal-softmax-probability-1/legalcount), range width<=.01;
  differences>.01 in original score cannot invert. Reorder ONLY quiet slots;
  capture/promotion/abs(score)>1 exact-terminal-range slots preserved.
  Interior ordering remains ORIGINAL baseline (not a hidden quiet-model mixture).
  theta0 delegates original search exactly; no feature overhead at theta0.
  positive search AST differs only root-order expression + injected Result/exception.

ROOT future launcher recipe (no actual clocks invented):
1. Copy actual original value-reanalysis registration per seed. Set NEW schema
   own-search-deeper-bestmove-reanalysis-registration-v1, NEW distinct7200 first/end,
   NEW output /dev/shm/harbichess-own-action-reanalysis/<seed>/labels-00001024.json.gz.
   Preserve ALL journal/config/native0/training/protected/selector/search prior pins.
   New closure.launcher=produce_actions.py SHA, executor=action_labels.py SHA;
   ADD closure.action_model=action_model.py SHA. Original other closure pins remain.
   Helper_source_commit still binds original five classical helper files, not a
   false claim that new producer code is in old history; ROOT separately pins
   newly committed producer code SHA/helper version. Source core remains6fcc.
   python produce_actions.py --registration <NEWregistered-json>
2. Optional reference_value_labels={'path':original exact RAM label path,
   'sha256':actualSHA} only AFTER original value labels are available/frozen.
   This compares ALL same1024 history/rawq float.hex/node/depth/eval/root packets.
   Missing old values are NOT a reproduction PASS. New action job does not touch
   active old value files, replace their schema, or reuse/extend their clock.
3. ROOT freezes labels SHA and producer-registrationSHA in training-protocol-DRAFT
   converted to registered math/source/helper/deadline-by-seed contract. For native
   qualification, create a distinct qualification600 protocol clock per seed:
   python qualify_actions.py --protocol <file> --protocol-sha256 <sha>
     --labels <newfrozen-labels> --source-repo <clean6fcc> --output <NEWdirectory>
     --seed <20262905|06> --cpu-core <ROOTchosen> --first <newfirst> --deadline <end>
   Executes whole8/pause4/freshprocessresume8, exact whole/split0/4/8 JSON bytes,
   then six fresh-process strict loads; stdout/stderr/failures retained. Owned
   child process groups receive only bounded cleanup; no old processes signaled.
   Qualification is NOT final64 or strength. This actual proof is STILL PENDING.
4. Newfit1800 protocol initializes NEW native0 fromzero with same frozen labels;
   do not pass qualification native as full resume under a changed contract.
   python train_actions.py --protocol... --labels... --output... --source-repo...
     --seed... --cpu-core... --deadline... (defaultstop64)
   --stop-at/--resume/--resume-sha256 offer exact SAMEcontract offline resume only.
   --audit-only --resume... provides strict fresh native0/final readonly load
   under its original phase guard. Expired TRAIN cannot be revived by this CLI.
5. Before games, ROOT separately runs new24 real-profile zero packet equivalence,
   bounded positive legality/budget/latency proof and freezes same160 development
   5arms + unchanged gates/audit900. Prospective labels7200/fit1800, not old resets.
   New real nodes/evaluation/profile/game proof is NOT performed here.

ROOT controls/reference policy: no model/view selection from observed heldout,
no MAX8 reset, development160 is not virgin confirmation. Same-engine prior and
E8 comparators use zero action weights. Capacity to achieve broad strength is a
hypothesis; learned root probabilities may be flat or harmful, epsilon may be
ineffectual, large8192 teacher's choices may be beyond512 inference reach.
Classical prior is hand-designed, so don't claim rules/model discovered fromzero.

Close prior art (offline known references, no web queries): Expert Iteration /
Anthony,Tian,Barber2017(arXiv1705.08439), AlphaZero own search imitation, Giraffe /
Lai2015(arXiv1509.01549), history/learned alpha-beta ordering. No novelty claim.

10 static/legal/gradient/native0-serialization/AST tests PASS; canonical RuffPASS.
No actual search calls, optimizer updates, fit, games, labels, network or Git.
Tests intentionally DO NOT claim real8/4/native qualification; ROOT must execute.
Initial6PASS1FAIL selector byte test preserved in verification note: formatter
changed local copy; immutable original recopied; actual active file untouched.
