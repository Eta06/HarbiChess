# UFUK compiler v2: validated host boundary and numerical graph

Before execution. V1 setup **failed** after2.4298s, before any parity case/timing:
Dynamo fullgraph rejects data-dependent Python branches in the existing action
bounds check (`torch.any(actions<0)` etc). The input validation was correct;
no inference-speed result, integration, fallback or success claim. Preserve
`artifacts/ufuk-compile-20261003/setup.json`/failure.txt and script hash.

Separate V2 tests the pairwise numeric `_features(inputs,actions)` graph only,
with identical dimensional/range guards executed by an explicit host wrapper
**before** each graph call. Guards stay in the timed boundary; malformed requests
remain refused. No changes to the active self-play code/model or default backend.
Fullgraph/dynamic Inductor still required for numerical body; no partial eager math.

Same frozen18f2aae5… weights, CPU FP32/oneTorch+compiler thread, existing4CPU/16GiB,
180s setup cap,6special-position cases×batch1/8/32,2e-5 parity/finite criteria.
Source/helper script hash and setup resource costs recorded. Failure is failed.
All timing/15%batch1 median gain/batch8-32<=10%regression criteria remain exactly
as in [UFUK-compile-preregistration-20261003.md](UFUK-compile-preregistration-20261003.md).
Setup may overlap actors; timed measurements wait until other work drains.
No claim of actor/game speed without a separately registered startup-inclusive test.
