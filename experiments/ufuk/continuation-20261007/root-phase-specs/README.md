Metadata-only wiring. No Torch/model/engine import, subprocess creation, training, search or registry execution. Existing frozen training/producer files remain unchanged. All first/deadline/operator values are ROOT supplied; every phase is checked before publication. Operator must match completed collection and cannot exceed1791448916.685839. Outputs publish once, only under work/harbichess or /dev/shm.

Before proof/fit, ROOT writes a tiny audit-set JSON:
```
{"schema":"NNUE-own-collection-replay-audit-set-v2",
 "helper":{"path":".../audit_collection_six_v2.py","sha256":"2082c379ac0dc0540e3e758ce70da09c20d972a267eee057a8b95cd2609a491e"},
 "audits":[{"seed":20262905,"result":{"path":"...","sha256":"..."},"clock":{"path":"...","sha256":"..."}},
           {"seed":20262906,"result":{"path":"...","sha256":"..."},"clock":{"path":"...","sha256":"..."}}]}
```
Both actual PASS results must have the corrected string-ID marker, exact six IDs at0/204/409/614/819/1023, source/registration/receipt/events/helper/clock SHAs, completion inside their clocks, and zero new rows/games/optimizer steps. Old known-bad1a5 helper cannot qualify. Manifest and EACH helper/result/clock ref are separately bound in prover inputs. ROOT also provides registrations.json mapping stringseed→exact completed producer registration path.

Converter: `specs.py --phase convert --registration REG --training FROZEN_TRAIN_DIR --producer FROZEN_PRODUCER_DIR --seed SEED --first FIRST --deadline END --operator-end OP --output NEW_SPEC_DIR`. It pins closed receipt/events/all sidecars and exact producer/features/prior source. `commands.json` gives the ROOT-only converter invocation; destination is exactly /dev/shm/harbichess-NNUE-own-data-v2/SEED. Two calls including converter; no result is generated for an unfinished collection.

Proof/fit preparation: same flags, plus `--phase proof` or `fit`, `--cpu-core CORE --audits AUDIT_SET --registrations BOTH_REG_MAP --teacher-registry ORIGINAL_TEACHER_PROFILE_DRAFT_OR_ACTUAL_V2_REG --teacher-admission CACHED_ADMISSION`. Original teacher registry must contain teachers[str(seed)] with original path/SHA refs, matching named collection parent. Fit additionally takes `--own-proof-result ACTUAL_OWN_PROOF/result.json --own-proof-contract ACTUAL_OWN_PROOF_CONTRACT`.

Preparation emits build-seal.json, registry-plan.json and commands.json. ROOT executes the frozen contracts builder command, then the supplied specs registry command, then the prover command emitted by that registry directory. This is four small CLI calls including preparation; metadata-only code cannot SHA-bind a nonexistent contract without bypassing the frozen strict/model-validation builder. ROOT may batch the latter three under its existing owner. No phase clock is re-stamped between calls. All helper-side validation/admission work is inside the same ROOT600proof/1800fit window.

Contract destinations: /dev/shm/harbichess-NNUE-own-contracts-v2/SEED/{proof,fit}.json. They deliberately sit outside the prover's publish-once output root. Final prover output is exactly /dev/shm/harbichess-NNUE-own-proof-v2/SEED or own-fit-v2/SEED; ROOT must ensure absent before original launch. Contract dataset/seed/mode/clocks are checked against the plan before run registry publication. Proof whole8/4/fresh8 +six strict loads eachseed and fit fresh64 +two eachseed remain frozen prover's responsibility, never fabricated by this factory.

Tests are synthetic metadata and require no planner/model calls. Positive audit fixture is not an actual qualification; real collection/qualification must come from ROOT's frozen helper. No strength or completed-fit claim is made here.
