# DEVIR micro arena — 2026-08-24

The accepted OCAK candidate `candidate-step-000040` from
`ocak-sanity-20260824-08` was evaluated against its unchanged random-initial
baseline. The arena runner source was commit
`bffa6782071f0e1417c7ce00a7e1e1cf1d5447f6`.

## Configuration

- 8 opening pairs, each played with reversed candidate colors
- 16 total games
- 4 uniformly sampled opening plies
- 4 MCTS simulations per move for both models
- 192-ply cap
- 16 parallel game workers with separate shared MLX inference queues
- Promotion minimum: 200 games with the 95% Elo lower bound above 0

The initial baseline was reconstructed from the OCAK run's recorded MLX seed
and network configuration, then saved as a checksummed arena artifact. Candidate
and baseline model hashes differ, confirming that the trained weights were
actually evaluated against the original baseline.

## Result

| Measurement | Result |
| --- | ---: |
| Candidate wins | 0 |
| Draws | 16 |
| Candidate losses | 0 |
| Candidate score | 50.0% |
| Estimated Elo delta | 0 |
| Promotion ready | No |
| Arena time | 9.94 seconds |

All 16 games ended by threefold repetition after an average of 70.56 plies.
Candidate color was perfectly balanced: eight games as White and eight as
Black, with four total points from each color.

## Decision

The candidate is not promoted and the champion pointer remains unchanged. The
arena shows no measurable playing-strength gain from the 40-step sanity pilot;
this is consistent with the pilot's purpose and tiny search/training budget.

The next candidate must restart from the unchanged baseline, not from this
failed candidate. Before another arena, HarbiChess should generate a larger
baseline self-play window with more search simulations and terminal variety,
then run a longer learner pilot while retaining the same validation, diversity,
finite-gradient, checkpoint, and promotion guardrails.
