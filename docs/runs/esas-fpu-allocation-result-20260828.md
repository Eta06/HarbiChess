# ESAS Parent-Relative FPU Allocation Result

Date: 2026-08-28  
Artifact: `artifacts/diagnostics/esas-fpu-system-teacher-20260828-01/result.json`

## Decision

The preregistered parent-relative FPU treatment failed. Keep the legacy zero-FPU
behavior as the default, do not tune the two treatment constants from this
result, and keep learner/latest, generation, and promotion disabled.

## Frozen treatment

- Root FPU reduction: `0.1`
- Interior FPU reduction: `0.2`
- Search budgets: `64`, `128`, `256`
- Opening pairs per arm: `32` (`64` games)
- Model, openings, worker count, and all other search settings matched the
  zero-FPU ESAS system-teacher run.

## Results

| Budget | W-D-L | Score | Bootstrap interval | Threefold | Tactical |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 3-61-0 | 52.34% | [50.00%, 55.47%] | 95.31% | 4/8 |
| 128 | 3-60-1 | 51.56% | [48.44%, 54.69%] | 93.75% | 6/8 |
| 256 | 10-54-0 | 57.81% | [53.91%, 62.50%] | 84.38% | 5/8 |

The treatment failed all three recorded reasons:

1. 128-search score did not exceed the preregistered 55% threshold.
2. The 256 tactical solve count regressed relative to 128.
3. The 256 arm lost `hanging-queen`, which the 128 arm solved.

Compared with the zero-FPU run, the 256 score rose only from 57.03% to 57.81%,
while 128 strength and tactical budget scaling worsened. This is not a stable
teacher improvement.

## Throughput

- Wall clock: 604.30 s
- Evaluated positions: 776,063
- Throughput: 1,284.24 positions/s
- Mean inference batch: 9.70

Throughput stayed effectively equal to the zero-FPU reference (1,285.37
positions/s), so the rejection is behavioral rather than a performance
regression.

## Next branch

Do not tune FPU. Audit and test genuine Full Gumbel MuZero allocation with
completed-Q and interior-node selection as a separately preregistered mechanism;
the repository's existing root-only Gumbel helper must not be treated as that
algorithm.
