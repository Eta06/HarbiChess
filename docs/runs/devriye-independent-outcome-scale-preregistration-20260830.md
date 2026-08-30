# DEVRIYE independent outcome scale preregistration

## Diagnosis

`devriye-continuous-pilot-20260830-10` established that 768 searched train rows
repair policy transfer: on 192 unseen validation rows, step 20 improved CE from
2.93366 to 2.91967 and top-action agreement from 8.33% to 19.79%. It still
rolled back because WDL micro CE regressed by 0.01410.

The update contained five independent known-terminal games. Their 166 labeled
positions repeat those five game outcomes; row count therefore overstates value
sample independence. Only two of the five new outcomes matched the historical
outcome attached to their starting state. A changed policy may legitimately
change the result, so historical labels will not replace new outcomes, but five
Monte Carlo results are too small and correlated for a 50% fresh WDL batch.

## Frozen pilot

- Fresh seed: `2026083701`.
- Keep three updates, rolling window two, 768/192 Full Gumbel-256 policy targets,
  40 learner steps, batch 64, learning rate `1e-4`, and all gates unchanged.
- Generate 96 latest-network Full Gumbel-64 continuation games per update from
  96 distinct historical games: 32 opening, 32 middlegame, and 32 endgame starts.
- Use 24 self-play workers, the already measured production throughput winner.
  Keep 96 additional plies, temperature schedule, and policy semantics unchanged.
- Require at least 24 real terminal games before learning. Max-ply rows remain
  unknown and cannot satisfy this floor.
- Keep historical/fresh WDL exposure at 32/32 and retain fixed fresh sampling of
  8 loss / 16 draw / 8 win, then game-balanced within outcome.
- Record unique terminal-game count, outcome/game distribution, phase, replay
  checksum, and teacher checkpoint provenance for every update.

## Decision

No threshold may be changed after results. Only all three accepted updates plus
the existing final WDL, continuation, tactical, and search-strength chain gates
authorize continuous production integration. Release promotion remains a
separate later decision.
