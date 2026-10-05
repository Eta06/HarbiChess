Separate CPU contingency; not an A100 qualification or historical result repair.

Source is 3be5b87db27a0fbde83464e7ea7157f0d9a76ae4, clean producer checkout; Torch must be exactly 2.14.1+cpu. Native schema is torch-search-acting-native-cpu-v3 and ledger v4. The development E1 and formal nine-checkpoint auditor use the SAME copied CPU production core. No actual CPU inference/training qualification has been performed by this preparation task.

Required root steps before training:
1. Freeze the separate protocol, original common firstclock, phase budgets, six neural epochs [1,2,3,5,7,8], exact helper/input SHAs, and statistical ledger allocation within existing MAX8; no ninth undisclosed qualifying family.
2. Run actual source3be CPU CLI whole2 / pause1 / fresh resume2, compare all six payloads at native0,1,2 and both journals, and strict-load both full native trajectories. This qualification is independent of A10095, the unknown A100 CLI receipt and failed earlier profiles. A tiny CLI does not validate the 64x256 witness collection.
3. Qualify actual CPU 64x256 E1 through cpu_contingency_qualify_e1.py. It requires all 16384 legal/action/terminal/UNKNOWN rows, original64-actor neural batch/masks, K8/18 dedup roots, actual six corruption checks, winning+visited-loss certificates, nonzero retained updates and changed model storage. It fails if data lacks any required witness. It never reruns or invents missing witnesses.
4. Keep ALL epoch0..8 native payloads and ALL journals; run cpu_contingency_full_audit.py incrementally while training. Do not clear native archives to meet disk requirements. Check actual file-byte growth during qualification before admitting both production seeds.
5. Exact independent fresh CLI epoch1->2 replay per seed remains required, followed by model parity, quiescent latency, fixed paired-color full strength schedule and unchanged gates. No early epoch selection.

All commands must use the clean SOURCE3be checkout via PYTHONPATH=CHECKOUT/src:THIS_ADAPTER_FOLDER, exact original config/input paths and OMP_NUM_THREADS=1/MKL_NUM_THREADS=1/OPENBLAS_NUM_THREADS=1. Start actor/training process and read-only auditor as separately owned groups. The helpers do not inherit a source8 CUDA success marker. Root controls outer hard06UTC and all process-owned cleanup. Outputs use atomic publish-once.

Full audit command:
PYTHONPATH="$CHECKOUT/src:$ADAPTER" "$PYTHON" "$ADAPTER/cpu_contingency_full_audit.py" --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" --run "$RUN" --output "$AUDIT/full-audit-result.json" --deadline-epoch "$AUDIT_DEADLINE"

Full audit manifest schema ufuk-cpu-contingency-audit-manifest-v1, status frozen-before-formal-execution, protocol_id cpu-contingency-v1; source_commit SOURCE3be; fixed_epochs8; config frozen CPU/64/256/new seed; producer_checkout; run; four inputs {initial_weights,book,experiment_config,protocol} with path+sha256; helper_sha256 EXACT six entries cpu_contingency_{adapter_controls,audit_support,audit_core,full_audit,fresh_cli_replay,qualify_e1}.py; neural_audit_epochs [1,2,3,5,7,8]; neural_witness_K8; original_training_started_epoch, original_training_deadline_epoch=first+whole_training_seconds, whole_training_seconds, whole_audit_seconds, absolute_audit_cutoff_epoch<=1791180000. Passed deadline CLI argument MUST equal min(first+whole_audit_seconds,absolute_audit_cutoff_epoch). No future clock or restart.

E1 audit command:
PYTHONPATH="$CHECKOUT/src:$ADAPTER" "$PYTHON" "$ADAPTER/cpu_contingency_qualify_e1.py" --manifest "$E1_MANIFEST" --manifest-sha256 "$E1_MANIFEST_SHA" --output "$E1_AUDIT_RESULT" --deadline-epoch "$E1_DEADLINE"

E1 manifest schema ufuk-cpu-contingency-E1-auditor-development-v1; scope cpu-contingency-development-E1-not-strength; run; checkout; source_commit; original_profile_deadline_epoch matching run metadata; started_epoch+whole_seconds=deadline (<06UTC); whole_seconds<=900; frozen_config; four inputs; helper_sha256 including qualifier itself and all imported local helpers; neural_witness_K8. E1 metadata max_epochs=checkpoint_interval=1. All original producer input hashes remain unchanged.

Fresh replay command:
PYTHONPATH="$CHECKOUT/src:$ADAPTER" "$PYTHON" "$ADAPTER/cpu_contingency_fresh_cli_replay.py" --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" --run "$RUN" --audit-run "$SIBLING_FRESH_REPLAY_RUN" --deadline-epoch "$REPLAY_DEADLINE"

Fresh replay output is an audit-only fork with one duplicate16384-row epoch, no new production selection. <=600-second actual clock, CPU-native-v3 all six payloads/journal exact; original production absolute deadline never changes.

Feasibility limits: observed free disk1188548608 bytes, quota400000/100000=4 CPUs. Source model itself294764 bytes; archived full-game journals and actor histories, not model weights, dominate uncertain disk cost. The support guard uses15GiB memory and256MiB disk minimum. These are prospective CPU resource safeguards; they do not establish that all native archives fit. Missing qualification, all9 audit, full576 games, unchanged strength gates, or hard06 means INCOMPLETE.


Prospective memory-accounting amendment (before CPU E1): memory.current includes reclaimable file cache. The operative15GiB metric is ESTIMATED nonreclaimable cgroup charge=max(0,current−(file−shmem)−slab_reclaimable); it is not RSS and not the conventional current−inactive_file working-set metric. Each failure records all component counters, configured cgroupmax, current−inactive_file diagnostic, current, and named violations. TOTAL current charge >16GiB is independently rejected; physical cgroupmax16GiB remains unchanged. File/shmem/slab counters are read-only. No cache deletion/dropCaches calls. Baseline/final arms use the same metric and total16GiB safeguard. memory.stat/current snapshots are separate reads and may fluctuate; a guard failure remains a failure, no after-data threshold changes.

CPU eligibility manifest (not audit manifest): qualification_ledger_slot8, source_commitSOURCE3be, fixed_epochs8, audit_helper_sha256=SHA(cpu_contingency_full_audit.py), replay_helper_sha256=SHA(cpu_contingency_fresh_cli_replay.py); seeds EXACT ordered61925/61926, each {seed,run,full_search_acting_audit:{path,sha256},fresh_search_acting_replay:{path,sha256}}. CLI cpu_contingency_eligibility.py --manifest FILE --output FILE; receipt schema ufuk-cpu-contingency-fixed-candidate-eligibility-v1. Its import is local cpu_contingency_adapter_controls, not any old own8 package.

Transitive source binding: producer dependencies are bound by exact clean3be source. ALL local CPU helper .py files (excluding tests) must be staged together and inventoried. Audit manifest requires the exact six audit closure modules named above. Strength Q helper_sha256 should include the entire local CPU runtime closure; bind_cli now verifies every declared helper before any main operation, not only caller. Do not rely on oldA100 import search paths. Source-only --help imports pass without training for auditor, E1 qualifier, replay and eligibility. The statistical functions and require_strength_gates AST remain identical to frozen own8 originals.
