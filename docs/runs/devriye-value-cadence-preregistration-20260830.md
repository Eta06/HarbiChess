# DEVRIYE value checkpoint cadence preregistration

## Evidence

On the exact cached `-12` update-1 data, per-step validation showed:

| Step | WDL micro CE | Macro CE | Pearson |
|---:|---:|---:|---:|
| 0 | 0.912848 | 0.937718 | 0.452612 |
| 1 | 0.915287 | 0.936241 | 0.452388 |
| 3 | 0.919957 | 0.933999 | 0.451781 |
| 7 | 0.923401 | 0.933904 | 0.450950 |
| 10 | 0.926570 | 0.935251 | 0.450192 |

Step 1 satisfies every existing WDL gate; the prior 10-step cadence never made
that state selectable. Policy requires later updates and is independently
selectable after the head-wise checkpoint change.

## Frozen change

- Keep all data, search, replay, optimizer, learning-rate, exposure, and gate
  thresholds unchanged.
- Evaluate WDL after every local learner step only until the first checkpoint
  satisfying all existing WDL gates is found; snapshot that earliest state.
- Continue evaluating policy at the existing 10-step cadence and select its
  earliest passing state.
- Compose the two heads, reset incompatible shared Adam moments, recompute both
  validation metric sets, then run unchanged material, continuation, tactical,
  and arena gates.
- First validate the mechanism against cached `-12` update-1 inputs. Cached
  evidence cannot authorize a generation. If all joint gates pass, run the
  complete fresh three-update pilot with seed `2026083901`.

## Decision

Failure cannot be rescued by changing cadence, thresholds, duration, or replay
weights after results. Only the subsequent fresh full-chain pass can authorize
production continuous integration.
