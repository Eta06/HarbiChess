# UFUK search signal diagnostic: before measurement

Both AYNA all-parameter and UFUK frozen-value policy self-learning failed their
registered strength gates. More updates alone are not the next controlled test.
Ask whether the best UFUK bootstrap's own search improves its actions, and how
much finite-budget value error still limits it after native-reference training.
This is a diagnostic, not a training run or a strength qualification.

Use unchanged `evaluation.teacher_probe` on the same96 full-history development
positions previously frozen by AYNA from the MERCEK16-simulation arena. Freeze
again from its unchanged source and require panel SHA to match AYNA metadata.
Eight historical development families, not a new test set or new Elo evidence.
Selection was by fixed game-history fractions, not reference/model quality.

Baseline: UFUK native-validation-selected step9250, SHA
`18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae`.
Latest raw-policy comparison: UFUK failed self-play generation3 SHA
`d5f5bbc5e43bf28ee06d558ee8f78242294ce0f9d1f17995835a52a159cf0de5`.
All searched arms use **baseline** policy/value: raw, FullGumbel16,128, and16
with same policy but Stockfish1024-node leaf value. Latest search is not measured.
Keep fixed existing Gumbel0/value_scale0.1/maxvisit_init50 and seed20261003.
Reference choices4096/32768, each distinct candidate forced at32768 nodes;
score/regret is native STM expected WDL and restricted to these tested candidates.
Fresh TT per query, Stockfish19/one thread/16MiB, record actual nodes and coherent
unbounded packets. Reference/leaf compute unequal; oracle is diagnostic only.

Hypothesis: replacing learned value reduces restricted regret by≥0.05 with
paired-family bootstrap lower>0, despite the native validation improvement.
Search128 vs16 uses the same≥0.05/lower>0 mechanism threshold. Also report raw
vs16, latest-raw vsbaseline-raw, exact teacher rank/top16 coverage, reference
budget agreement and value MAE. Eight-family conditional bootstrap is fragile;
report conservative bounds and never call this general chess strength. No
threshold changes, arena reuse for promotion, or optimizer updates.

Existing CPU allocation only, one Torch/engine thread,900s wall ceiling,15s query
watchdogs. Stop on incomplete/bounded/nonfinite/protocol/resource failure; retain
every query/result and report incomplete, not missing-position success. Require96
completed positions, exact frozen panel hash, all choices/legal histories audited.
Keep best bootstrap and all failed checkpoints; no new paid resources.

The original UFUK arena source-integrity reproduction is a separate fixed test
and is not incorporated as additional independent strength data. Log final
diagnostic evidence, actual resources and limits in Git/Space. Release upload
remains blocked; locally preserved binary weights are not a completed backup.
