# AKTARIM train-only fit result (2026-08-28)

## Decision

No preregistered arm passed every train-fit gate. Fresh validation, search
qualification, arena, generation, and promotion were not started.

## Matrix result at step 480

| Rank | LR | Teacher rho | Verified gain lower 95% | Harmful | Regret |
|---:|---:|---:|---:|---:|---:|
| 8 | 2e-4 | 0.3793 | +0.0049 | 12.40% | 0.0972 |
| 8 | 1e-3 | 0.4653 | +0.0225 | 10.55% | 0.0777 |
| 32 | 2e-4 | 0.3961 | +0.0112 | 12.40% | 0.0885 |
| 32 | 1e-3 | 0.5818 | +0.0276 | 11.35% | 0.0724 |

All arms missed the unchanged 10% harmful limit. The stronger optimizer/rank
clearly improves imitation and verified gain, but does not qualify.

The inherited “5% cross-entropy improvement” gate was also proven impossible
for this target. SIPER constrains `KL(target || raw)` to 0.10, which is the
maximum reducible cross-entropy gap. Against baseline CE near 3.139, even exact
imitation can improve only about 3.2%. Existing experiments remain failed; a
future preregistration should measure fraction of reducible KL gap rather than
retroactively changing their outcome.

The next design decision is a train-only model-update trust region: fit the
qualified SIPER distribution, then scale/project the learned policy delta using
only training safety metrics before one fresh validation test. This is preferred
to more data, longer training, or threshold relaxation.

## Frozen artifacts

- `artifacts/diagnostics/aktarim-r8-lr2e4-20260828-01/result.json`
- `artifacts/diagnostics/aktarim-r8-lr1e3-20260828-01/result.json`
- `artifacts/diagnostics/aktarim-r32-lr2e4-20260828-01/result.json`
- `artifacts/diagnostics/aktarim-r32-lr1e3-20260828-01/result.json`
