KINGBUCKET NNUE16 — PREPARED PROTOTYPE, NOT REAL TEACHER/OWN TRAINED

Geometry:16 mover-oriented ownking2x2-square buckets x12 relative piece types
x64 oriented squares=12288 embedding rows; hidden16 tanh; scalar head16+bias.
196625 float64 trainable parameters. Both kings/12disjoint piecebitboards, at
most32pieces, fixed relativepiece then oriented-square accumulation order.
Not incremental NNUE accumulators yet. Shared 2x2 king buckets reduce king
resolution; hidden16 limits interaction capacity. Future actual24latency and
512search strength must decide feasibility, not synthetic timings.

Prior remains exact frozen humanprior18. C computes ONLY residual. Literalzero
head delegates original Python prior nonterminal verbatim, including builtin
compensated sum and math.tanh. Nonzero uses authoritative Python prior-logit
plus C residual. No C prior reconstruction, no atanh roundtrip. Zero delegation
synthetic signedzero and compensated-sum cancellation tests pass.

SOURCE/SCHEMA
model.py sparse float64 EmbeddingBag; native.py full CPU native; evaluator.py
pinned AuthoritativePrior adapter+compiled residual; forward.c/build.py explicit
no-fast-math/no-FMA extension; train.py versioned CLI; qualify.py syntheticproof.
Full native own-kingbucket-nnue16-full-native-cpu-v1 stores current+immutable
baseline models, full Adam3states/counters, global Python/TorchCPU RNG, private
sampler RNG and complete contract. All196625 parameters used; no old material
unused-readout parameters copied. Full native actualsynthetic6,311,475B<8MiB.
Storage comparisons require exactdtype/shape/bytes including signedzero.
No TorchZIP serialization-byte identity claim; eachartifactSHA separatelyvalid.
Only SHA-sealed local artifacts may enter torch.load(weights_only=False).
Public archives must be rawbyte-verified first; untrusted pickle not safe parser.

PHASES FIXED BEFORE FUTURE ROOT REGISTRATION (NO REAL RUN HERE)
1 teacher-bootstrap: zero additive head on randomhidden std.01seed; Adam lr.001,
   betas.9/.999,eps1e-8,decay0,b256,clip5,256updates,MSE(tanh(priorlogit+residual),y).
   Future labels ONLY source-fixed OWN CLASSIC TRAIN roots, exact fullhistories,
   SF8192 cp targets tanh(cp/600), moverPOV; exactmate sign. These are proxy
   values, NOT calibrated WDL and NOT teacherweights. No VAL/knownbook roots,
   no endpoint/seed/hyperparameter sweep. ROOT must produce/freeze labels later.
2 own-learning: same architecture/search/source, named teacher candidate SHA
   imported as FULL weights-only initializer; NEWAdam0 and NEWglobal/samplerRNG,
   own64updates lr.001/b256/clip5, frozen humanprior fixed. Baseline snapshot is
   the exact teacher model and source-qualified candidate; kept bitimmutable.
   Targets from fresh OWN8192/q2/max8 search and/or separate truecompleted-ownplay
   labels only under an explicit prospectively frozen target-dataset version.
   This prototype's MSE supports ONE declared bounded scalar per row. Mixing
   ownQ/terminal losses is a future newobjective, not silently inferred here.
   Frozen BOOTSTRAP versus learned SAMEsearch and bothseeds mandatory; teacher
   strength alone is not selflearning evidence. Real ownplay/unknown handling
   and immutable terminal outcomes remain required for later strength admission.
3 full resume: state restores current+baseline+Adam+all RNG/counter/contract.
   It cannot also take a bootstrap input. Ownphase restore rechecks external
   bootstrap candidateSHA and baselinebits; baseline source required to restore.

PRODUCTION CONTRACT/DATA GEOMETRY
contract phase/updates/seed/math/feature_schema/dataset_sha256/source_sha256
(training model/native/train exactabsolute paths), core_source_repo/commit,
original_first_epoch/deadline<=1791273600. Non-synthetic additionally requires
prior_helper_path/SHA, target_provenance_path/SHA, inference_source_sha256 map
of evaluator/C/binary/priorhelper +producer/protocol/source provenance. Ownphase
requires bootstrap_candidate_path/SHA. Data schema own-kingbucket-sparse-training-
data-v1,phase,rows(indices,prior_logit,target). Fullhistory/legality/protected
exclusions and exact prior/target extraction must be frozen in the FUTURE ROOT
label producer; this prototype trainer verifies sealed dataset/source, not a
complete producer. No registered actual dataset currently exists.

CLI (ROOT FUTURE EXECUTION ONLY)
python train.py --contract <frozenJSON> --dataset <frozenJSON> --output <NEW>
 --stop256 [teacher]; --stop64 --bootstrap-candidate <teacher.pt> [own].
Fullresume --resume <native.pt> --resume-sha256 <SHA>, sameoriginal contract.
--audit-only performs strict load with no optimizer; oldexpiredTRAIN clock cannot
be reused for audit: separate read-only prospective audit wrapper/version needed.
Resources CPU1/cgroup15GiB/diskfloor256MiB/native8MiB, immutablefirst/end.
Actual train/qualify --help imports/argparse are provided; ROOTsourcepin later.

VERIFIED LOCAL ONLY
Synthetic whole8/pause4/fresh8 +six strict freshinterpreter loads PASS22.8s,
all native payload tensor/model/Adam/RNG bits exact, recordsRAM
/dev/shm/harbichess-kingbucket-synthetic-native-v1/result.json.
24synthetic bitboard packets (bothmovers) Cnonzero residual<=1e-12 vsTorch;
zeroC+prior literalHEX delegated; geometry,baseline,Adam/RNG/counter mutation
rejections; weights-only NEWAdam bridge tested. Never actuallegal-board neural
inference, SF labelquery, ownsearch/game, teacher/realdata SGD or GPU/network.
Build binary ABI Python3.12/LinuxCPU; ROOTmust rebuild/seal ifruntimechanges.
Realfullhistory/specialcase/inputtrace+Cparity and nativefreshprocess on actual
source/dataset are future qualification gates; synthetic proof is insufficient.

Closest prior art: NNUE/HalfKP-style sparse king-piece features, supervised
engine-score distillation and ownsearch ExpertIteration/AlphaZero family are
established. No originality, breakthrough or strength claim.
