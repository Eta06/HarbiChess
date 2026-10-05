# UFUK-QSEARCH: fixed160-game result, 2026-10-05

Both seed development strength screens **FAIL**. No formal admission, checkpoint promotion, self-learning success, or novelty claim. Original teacher-origin e8 remains the strongest independently validated checkpoint.

This run reused the fixed VALUE step2048 critics and their zero-rebased controls. It performed no new training, teacher queries, or self-play data generation. Original e8, untrained, and trained models all used identical all-legal-root alpha-beta, 512 recursive nodes, and two-ply quiescence. MVV-LVA only orders moves. A search change alone cannot count as learning.

| Seed | Trained vs same-search e8 | Trained vs same-search zero | Trained vs SF512 | e8 vs SF512 | Zero vs SF512 | Paired gain vs e8 |
|---|---:|---:|---:|---:|---:|---:|
| 20262505 | 0.25 | 0.9375 | 0.0625 | 0.0 | 0.0625 | 0.0625 |
| 20262506 | 0.34375 | 1.0 | 0.0 | 0.0 | 0.0625 | 0.0 |

All160 games independently replayed: 10,246 legal continuation plies, 1,339,199 actual Stockfish nodes, 3,908,597 counted neural-search nodes; UNKNOWN caps0 in every16-game arm. Exact seed/arm set, pairing, result viewpoint, terminal history, model/helper provenance, and per-move NN/SF receipt coverage passed. Neural root counts match every legal-root count and NN budgets stay ≤512. Input artifacts remained unchanged.

The unchanged criteria require both direct scores>.60, paired SF gain>.10, final SF score≥.25, caps≤.05, trained-vs-zero>.60 and positive SF gain over zero. Both direct, main SF-gain, final SF and zero-relative SF-gain gates fail. Strong wins against the zero-value control demonstrate a learning effect relative to that weak control; they do not demonstrate strength gain over e8 or satisfy the objective.

Eight known opening families/color pairs, not virgin confirmation. Original e8 and rebased deterministic trajectories repeat across the two seeds, so these are not independent new control families. The recorded paired95% bootstrap/Hoeffding intervals are descriptive; the formal adjusted98.75%, latency and closedMAX8 protocol remains unchanged. Stockfish actual nodes can exceed its requested512 stopping budget; budgets and wall times are separately recorded, not claimed as equal compute.

The q-depth boundary uses static value even in check; the search qualification does not promise exhaustive check evasion. The full rule/history replay validates legality, not every NN-search calculation; a separate bounded independent chronological re-search review is being performed.

Original start1791213839.617308 and whole72000s clocks were preserved through the CPU connection outage. The run completed without restart. The new user deadline2026-10-06T08:00:00Z is enforced by a separately tested exact-owner PID/startticks guardian; it does not reset historical clocks. No GPU/SSH or paid compute used.

Interpretation: the affine own-policy value generalizes internally and beats zero, but these outcomes do not overcome the e8/SF controls. Own-policy returns estimate the behavior policy value, not unchosen-action or optimal values. Next preregistered test is the lower effective-capacity spatial-shrinkage critic, followed by a fixed-e8 residual control if needed. Standard alpha-beta, shrinkage, residual learning and reanalysis are not claimed as original contributions.

Evidence: [protocol](../../experiments/ufuk/cpu-budget-search-v1/protocol.json), [independent fullhistory result](../../experiments/ufuk/cpu-budget-search-v1/actual-result/arena-fullhistory-result.json), [original owner clocks](../../experiments/ufuk/cpu-budget-search-v1/actual-result/arena-owner-result.json). Full game JSON/PGN and model/native artifacts remain locally preserved; CPU public Release backup is still PENDING, separate from the older128 verified assets.
