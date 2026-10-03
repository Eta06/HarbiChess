# UFUK value experiment: preserved reporting failure and full-state v2 migration

Before any candidate update. Input preparation passed:10000native anchor rows/
36families;2663own training rows/84terminal games;704heldout rows/22terminal
games/11families. All unknown games excluded, own train/current-heldout overlap0.
Input cache/metadata remain unchanged and common to both arms.

Source66a5e1bd21e4b90792f96eb34aa43c98442369ed anchor paused successfully250,
then fresh-process full native optimizer/RNG/sample cursor resume reached1000.
Best heldout CE remained initial1.32684233647, with native retention passing;
registered1000step patience stopped computation. The CLI then exited1 because
the immutable `publish_json` helper refused to overwrite the existing result.json.
This is a failed result-publication process, not a successful completed CLI.
The250summary,1000checkpoint, full sample trace, failed command/stdout/stderr and
leftover result.json.partial remain preserved. No additional anchor update is
authorized after its registered stop; no failed pipeline is silently relabelled.

Reportingv2 stores an immutable session-step-N.json and atomically refreshes a
latest result pointer. It also refuses to resume past an already reached patience
or retention stop. Numerical loss, sampling, optimizer, model, cache and gates
are unchanged. Core full resume schema/encoder/action/replay/model remain1.

Explicit migration CLI `python -m harbichess.training.ufuk_value_outcomes
migrate-reporting-v1 DESTINATION CHECKPOINT CACHE` creates a fresh reporting-v2
run, never edits original files. Only recorded moduleV1 SHA
15c1cbbdca2ce56be2ea50fb82ab9acd1fc81dba77102047e8af0710799cb8de is accepted;
all other numerical dependencies/cache hashes and exact sample trace must match.
Copies latest and selected-best complete checkpoints, preserves all model
tensors/optimizer/Torch sampling RNG/cursor/input-cache links; new config marks
reporting_schema2 and actual packaging source. Model file hashes may change only
through provenance metadata, with parameter tensors unchanged. No training
updates occur in migration. Actual tests compare the entire native optimizer/RNG
state and model tensors bitwise. This is full-state migration, not weights transfer.

Publish the stopped control from its migrated1000checkpoint without another
update. Start the candidate once from the original18f weights on the same cache,
same seed/sampler/loss/budget. Compare common-step sample and TorchRNG hashes
against the preserved source66 control. Record source difference explicitly;
reporting fix supplies no additional statistical samples or numerical training
change. Native anchors are regularization, so any eventual outcome cannot be
described as pure self-play from scratch or proof of a sole value bottleneck.

The earlier new-module filename collision with historical scalar calibration
caused8collection errors before training. Both historical files were restored
byte-for-byte, the new experiment uses a separate ufuk_value_outcomes module,
and507full tests/zero skips passed before the source66 run. Those failed logs
are preserved too. No history, checkpoints or unsuccessful gates are removed.
