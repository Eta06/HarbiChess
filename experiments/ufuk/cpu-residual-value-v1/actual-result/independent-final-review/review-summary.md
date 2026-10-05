# Independent residual arena audit

**Result:** all 160 recorded games pass independent full-history legality and terminal replay. The registered development strength screen fails for both seeds. This is not a strength promotion or virgin-book result.

## Verified scope

- All 10 tournament records / 160 games and all 11,568 continuation plies were independently replayed from the initial position, including each prescribed opening prefix, every legal move, terminal outcome, cap state, recorded search move, neural and Stockfish node record, and per-game summary.
- Tournament contracts, protocol, opening book, all model files, and inference/search helper hashes were checked. The source checkout is clean at `6fcc8b476d25495d1c9c413e55b2c7ba4794013e`.
- Six fixed neural search packets were rerun on CPU with one thread: two each for E8, zero-residual rebased, and trained residual, at the first two candidate moves of the first fixed opening for seed 20262705. Every move, value, node count, evaluation count, depth, and root-action count matched. For residual, the loaded model was schema 2 / `additive-v1`; the tested adapter called the full `masked_policy_value` path so the inherited E8 WDL logits and residual were composed once.
- No games, training updates, or new strength queries were created.

## Replayed totals

| Measure | Independent count |
|---|---:|
| Games | 160 |
| Continuation plies | 11,568 |
| Actual Stockfish nodes | 1,540,747 |
| Stockfish moves above requested 512 nodes | 884 (maximum 598) |
| Neural nodes | 4,383,812 |
| Neural evaluations | 4,008,328 |
| Max-ply caps | 0 |

## Registered gates

| Seed | Residual vs E8 | Residual vs rebased | Residual vs SF512 | Outcome |
|---|---:|---:|---:|---|
| 20262705 | 0.09375 | 0.09375 | 0.03125 (0W/1D/15L) | FAIL |
| 20262706 | 0.15625 | 0.15625 | 0.03125 (0W/1D/15L) | FAIL |

Both seeds miss the direct-over-E8 `>0.60`, SF gain-over-E8 `>0.10`, final SF `>=0.25`, and residual-over-rebased `>0.60` thresholds. The paired SF gain over the zero-residual rebased control is positive at 0.03125 for both; it does not satisfy the full gate.

## Audit clock and limits

The independent replay started at epoch `1791225920.4132109`, had a 600-second deadline at `1791226520.4132109`, and finished at `1791225927.512798`. Its fixed six searches were the only extra NN searches. Stockfish exceeded the requested 512 nodes on 884 moves, so node counts do not establish equal compute. The openings are the registered eight development families, not virgin confirmation positions.
