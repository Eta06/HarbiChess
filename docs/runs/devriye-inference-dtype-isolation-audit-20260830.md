# DEVRIYE inference dtype isolation audit

## Failed run

`devriye-continuous-pilot-20260830-01` correctly rolled update 1 back because auxiliary material
predictions and the frozen-parameter hash changed. All behavioral metrics otherwise met their frozen
per-update gates.

## Root cause

`MLXPolicyValueBackend` calls `network.set_dtype(mx.bfloat16)` when an inference evaluator is
constructed. DEVRIYE passed the live FP32 learner network directly into baseline tactical and arena
evaluation. That converted every parameter, including frozen trunk and auxiliary material weights,
to BF16 in place. The saved update-000 checkpoint confirms quantization deltas across every parameter;
the material head itself was never optimizer-trainable.

A direct 40-step gradient audit without inference construction changed only the preregistered policy
and invariant-WDL parameter prefixes.

## Correction

Search target production already uses a frozen previous-network clone. Tactical and arena evaluation
must also receive independent clones; the live learner network must never be handed to a component
that changes inference dtype. Add a regression test that an inference evaluation may quantize its
clone but leaves the live learner parameters and dtype exact.

The original compute, seeds, data, learning rate, update count, and all gates remain unchanged. The
failed result is retained. A fresh run ID may repeat the preregistered pilot only after dtype-isolation
tests pass.
