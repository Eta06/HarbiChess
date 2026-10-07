# Procedural legal-start own-learning v2 — unexecuted production route

This is a separate root-distribution experiment. The previous4096-position pool comes from146/135 CLASSIC own-training trajectories; seed05 has3166/4096 positions sharing e4/e5/Nc3/Qg5. These metadata motivate a diversity control, not a causal diagnosis or a strength claim.

Each bank selects the first4096 acceptable positions within16384 attempts. A dedicated seeded Python RNG first draws a length uniformly from6..24, then chooses uniformly from sorted legal UCI moves at each ply, starting at the standard root. Terminal roots, protected final conservative placement aliases and previously accepted final aliases are rejected. There is no model, Stockfish, outcome-quality, material-quality or winner filter. This is a conditional random-walk distribution, not uniform sampling of reachable chess states. Distinct procedural IDs are not independent played-game evidence.

Protection is FINAL ROOT only. Every interior alias, including common START, is archived in accepted rows and every attempt trace; interior protected aliases are explicitly reported. The existing collection rule excludes a whole episode if its CURRENT played positions or traced evaluator inputs hit protection. No claim is made that all ancestral prefix states are protected. A future virgin-book exclusion union must include the complete bank trace and collection exposures. Preserving full prefixes retains castling, en-passant, repetition and50-move context.

`root_bank.verify` independently regenerates all draws, accepted histories and rejected attempts from the bound seed/protection/source/runtime and compares exact serialized source/trace bytes. A capped failure retains the complete attempted trace and cannot auto-extend or switch seeds. Synthetic tests use tiny counts and are rejected by the production verifier.

## Runtime and native identity

Model, native, learner, prover, zero initializer and current-parent admission bytes remain identical to frozen human-prior-own-v1. Numerical native schema `human-prior-own-nnue16-native-cpu-v1` and own phase64 remain unchanged. New collection/build/receipt/conversion/replay/ledger protocol names are explicitly `human-randomstarts-...-v2`; contracts record `root_source_type=procedural-uniform-legal-walk-v2` and every new helper SHA. This is input/source provenance, not an old checkpoint relabel. Original strict full-native source/contract/RNG semantics remain; parent-to-child initialization is WEIGHTS ONLY with fresh Adam/global/private sampler RNG. Full resume is within one exact phase contract only.

Original de53 search only:8192 nodes/q2/depth8;128 roots×at most16 fresh plies;1024 completed root-search Q targets; fixed64 updates. UNKNOWN cap/tail rows may have independently completed root-search Q, but have no invented own-terminal WDL. Old teacher-label ancestry keys are removed. No fresh teacher labels are read. Initial parent must be literal zero or a previously proven own parent under its ORIGINAL admission/helper closure.

## ROOT execution recipe (no phases executed here)

All clocks and generation/seed declarations are ROOT supplied. Operator cap1791448916.685839. ROOT must prospectively freeze separate bank seed(s), exact protected int64 input, helpers, parent and phase clocks before generating a bank. Run through the existing ROOT owner/launch guardian with one CPU/15GiB/disk floor256MiB. No inherited phase deadline is rewritten.

1. For each seed, stamp a NEW<=600s bank clock and call:

```
PY prepare_bank.py --seed BANK_SEED --first FIRST --deadline END --operator-end-epoch OP --protected-aliases PROTECTED --output BANK_REG
PY root_bank.py --registration BANK_REG --output /dev/shm/harbichess-human-randomstarts-bank-v2/SEED
```

2. Reuse original current-parent admission seal/result (actual literalzero0 or proven own64) and original inference directory/C extension; stamp collection<=7200s:

```
PY prepare_collection.py --admission-seal PARENT_SEAL --admission-result PARENT_RESULT --inference-directory INFERENCE --extension EXT --procedural-bank-receipt BANK/receipt.json --protected-aliases PROTECTED --root-pool-output POOL --first FIRST --deadline END --operator-end-epoch OP --cpu-core CORE --output BUILD_SEAL
PY metadata_factory.py --seal BUILD_SEAL --output COLLECTION_REG
PY run_collection.py --registration COLLECTION_REG
```

`prepare_collection` only creates metadata. The factory fully verifies procedural source bytes; runtime repeats validation BEFORE model/search helper loading. Output is a NEW `/dev/shm/harbichess-human-randomstarts-ownq-v2/g-G/SEED` directory.

3. Both complete receipts/events/alias inventories must pass the NEW `audit_collection_six.py`, exactly six original string IDs at offsets0,204,409,614,819,1023 per seed, under ROOT-stamped<=600s audit clocks. Its CLI `--help` specifies exact clock/registration/receipt/events/output flags. Freeze a `human-randomstarts-own-replay-audit-set-v2` mapping both result refs; no v1 audit can be substituted.

4. `specs.py --phase convert --registration REG --seed SEED --training STAGE --producer STAGE --first FIRST --deadline END --operator-end OP --output RAM_SPECS` emits exact converter seal/commands, <=600s. Then `specs.py --phase proof` with the same inputs plus `--registrations BOTH_REG_MAP --audits BOTH_AUDIT_SET --cpu-core CORE` emits contract/plan. `contracts.py --seal ... --output ...` and `specs.py --phase registry --plan ... --output ...` emit the actual `prove.py --registration ...` recipe. Proof must run whole8/pause4/FRESHresume8 and all six actual fresh strict native opens per seed. No synthetic proof replaces it.

5. Only after actual proof, `specs.py --phase fit` supplies additionally `--own-proof-result RESULT --own-proof-contract CONTRACT`, a NEW<=1800s first/end and same parent/data/audits. Its emitted original prover orchestration runs fresh0→64 then two actual fresh strict opens. Use ledger only after all complete. Same source/search parent and child, unchanged profile/strength gates. Existing known160 adapter currently recognizes v1 collection identities: it needs an explicitly versioned source-distribution adapter before new arena admission; no strength readiness claimed here.

Giraffe (arXiv1509.01549) and random legal perturbation curricula are prior art. This is a data-distribution hypothesis, not novelty. Seven pure tests and16 CLI imports/help pass; no4096 root bank, model forward, search, fitting or games have been executed by this implementation agent.
