# FREN policy-delta projection result (2026-08-28)

## Decision

No global projection scale passed. Fresh validation, search qualification,
arena, generation, and promotion remain blocked.

## Evidence

The exact reducible target gap was 0.09895 CE. Scale 0.2 was the closest safe
candidate but closed only 15.33% of that gap and selected harmful actions on
10.29% of train rows. Scale 0.3 closed 21.86% but harmful selection was 10.55%.
All larger scales also exceeded the unchanged 10% harmful limit. No rounding or
post-result threshold adjustment was applied.

Global delta scaling cannot simultaneously satisfy target absorption and the
argmax safety boundary. The current global-linear policy representation does
not share the origin/move-plane geometry of the 4672-action encoding. A
parameter-shared spatial residual policy head is the next representation
hypothesis; it requires its own preregistered train-fit and fresh-validation
evidence.

## Frozen artifact

- `artifacts/diagnostics/fren-policy-projection-20260828-01/result.json`
