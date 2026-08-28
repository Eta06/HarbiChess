# ESAS system teacher qualification result (2026-08-28)

## Decision

The preregistered gate failed. Learner/latest, replay generation, arena
promotion, and release promotion remain blocked. Thresholds and sample sizes
were not changed after the result.

The result nevertheless changes the diagnosis: 256-simulation clean PUCT is
already stronger than the raw network in paired play, but the collapsed value
head makes lower-budget search weak and tactical behavior non-monotonic. The
next correction belongs in value representation/training, not another policy
target heuristic.

Artifact: `artifacts/diagnostics/esas-system-teacher-20260828-01/result.json`

## Frozen paired play

Each arm played 64 games from the same 32 eight-ply openings, once per color,
against the raw legal policy argmax.

| Teacher | W-D-L | Score | Paired 95% interval | Decisive score | Threefold |
|---|---:|---:|---:|---:|---:|
| 64 simulations | 2-62-0 | 51.56% | [50.00%, 53.91%] | 100% | 96.88% |
| 128 simulations | 3-61-0 | 52.34% | [50.00%, 55.47%] | 100% | 95.31% |
| 256 simulations | 9-55-0 | **57.03%** | **[53.13%, 60.94%]** | 100% | 85.94% |

Raw-versus-raw produced 64/64 threefold draws. No arm produced a max-ply
termination. Search strength therefore came from converting repetitive draws
to wins, not from hiding losses behind draws. The 256 arm individually clears
the frozen strength, interval, decisive, and behavior conditions. The full gate
still fails because 128 did not exceed 55%.

## Tactical behavior

| Policy/search | Solved |
|---|---:|
| Raw | 1/8 |
| 64 | 5/8 |
| 128 | 7/8 |
| 256 | 7/8 |

The aggregate 256 count did not fall, but `hanging-queen` was solved at 128 and
lost at 256 while another case improved. This violates the frozen no-regression
condition and repeats the earlier OMURGA allocation symptom.

The raw value prediction across all eight tactical cases was effectively
constant: approximately `-0.022` in every position. Search discovers mates and
captures by eventually reaching terminal nodes; non-terminal leaf values provide
almost no ranking signal. This explains both the large improvement at 256 and
the weak/non-monotonic lower-budget behavior.

## Runtime

- Wall time: 647.35 seconds
- MLX positions: 832,080
- End-to-end throughput: 1,285.37 positions/second
- Batches: 93,845; mean 8.87, largest 16
- Backend time: 211.34 seconds

The frozen run confirms that more search compute can manufacture useful play,
but it is an inefficient substitute for a discriminating value model. The next
experiment must improve dense value learning while retaining terminal WDL as the
main semantic target. Existing KOPRU replay contains 10,038 known-outcome rows
and 9,216 masked max-ply rows; its stored search root value has only 0.294
correlation with known outcomes. Any bootstrapped search value must therefore be
an explicit auxiliary target, not silently mixed into WDL.
