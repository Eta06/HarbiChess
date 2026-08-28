# ESAS Full Gumbel System-Teacher Result

Date: 2026-08-28  
Artifact: `artifacts/diagnostics/esas-full-gumbel-system-teacher-20260828-01/result.json`

## Decision

The preregistered Full Gumbel system teacher passed every frozen gate. This is
the first ESAS search mechanism that qualifies as a materially stronger teacher
than the raw network. A small frozen learner-transfer experiment is authorized;
new large replay generation and promotion remain unauthorized.

## Frozen result

| Budget | W-D-L | Score | Paired interval | Decisive score | Max-ply | Threefold | Tactical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 32-32-0 | 75.00% | [68.75%, 81.25%] | 100% | 48.44% | 1.56% | 4/8 |
| 128 | 33-31-0 | 75.78% | [69.53%, 82.03%] | 100% | 17.19% | 28.12% | 4/8 |
| 256 | 52-12-0 | 90.63% | [85.16%, 95.31%] | 100% | 9.38% | 9.38% | 4/8 |

Raw control played 64/64 threefold draws and solved 1/8 tactical cases. The
256 arm won as both colors (25 white wins and 27 black wins), so the result is
not a color-side artifact. No budget lost a tactical case solved by the prior
budget.

All preregistered gates passed without changing thresholds, sample size, or
mechanism constants after the run.

## Mechanism interpretation

The causal result rejects the previous assumption that the collapsed neural
value makes every low-budget search teacher unusable. Allocation was a major
part of the failure: with the same network and simulation count, shared-tree
Sequential Halving plus interior completed-Q visitation converted shallow leaf
evidence into much stronger play than PUCT and parent-relative FPU.

This does not prove the value head is healthy. Full Gumbel's formal improvement
claim assumes correctly evaluated action values, and the earlier ESAS value
diagnostics disproved that assumption for this model. The observed system-level
strength is empirical authorization for controlled transfer, not permission to
skip value/calibration gates.

The 64 arm's 48.44% max-ply rate is also a warning against choosing the cheapest
budget from score alone. The learner-transfer pilot should use the qualified
256 teacher and preserve unknown max-ply value masks.

## Performance

- Wall clock: 2,269.87 s
- Evaluated positions: 1,710,708
- Throughput: 753.66 positions/s
- Mean inference batch: 9.50
- MLX backend time: 673.94 s

Legacy zero-FPU PUCT reached 1,285.37 positions/s on its matched qualification.
The current Python Full Gumbel implementation is therefore about 41.4% slower
in positions/s. The quality result is retained, but this implementation needs a
separate bitwise/decision-equivalent hot-path optimization before large-scale
generation.

## Authorization boundary

- Full Gumbel target semantics/provenance checks: authorized.
- Small fixed-compute learner-transfer ablation: authorized after those checks.
- Large replay generation: blocked.
- Promotion/release champion update: blocked.
