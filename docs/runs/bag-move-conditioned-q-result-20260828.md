# BAG move-conditioned Q transfer result (2026-08-28)

## Decision

BAG failed its frozen learner-transfer gate. No candidate, completed-Q search,
arena, generation, or promotion is authorized.

## Teacher evidence

The fresh raw 512/800 Q set still failed the preregistered 75% top-two overlap
gate (train 71.88%, validation 67.71%). The uncertainty-preserving label gate
passed after excluding unstable action mass: validation stable visit mass was
84.53%, stable-Q/verifier Spearman was 0.4029, conservative verified gain had a
95% interval of +0.0300 to +0.1149, harmful ratio was 4.17%, and mean regret was
0.0409.

## Learner result

The destination-aware head preserved policy and WDL logits exactly and trained
for the preregistered 480 steps. It did not transfer the qualified teacher
signal. At step 60, teacher-Q Spearman was only 0.1267, the verified-gain 95%
interval was -0.0414 to +0.0059, harmful action selection was 20.83%, and mean
verified regret was 0.1234. Later checkpoints were no better. Validation Q MSE
also never reached the required 20% reduction.

This falsifies the narrower hypothesis that missing destination geometry was
the main transfer blocker. A small frozen-trunk auxiliary Q head can reduce its
weighted regression loss without producing reliable action ranking or verified
improvement. The next experiment must transfer uncertainty directly into the
deployed policy representation rather than treating a separate Q head as the
learning endpoint.

## Frozen artifacts

- `artifacts/diagnostics/bag-raw-action-value-dataset-20260828-01/dataset.json`
- `artifacts/diagnostics/bag-uncertainty-labels-20260828-01/labels.json`
- `artifacts/runs/bag-move-q-20260828-01/result.json`
