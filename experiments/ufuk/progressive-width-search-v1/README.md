# Progressive-width search v1 — source-only development prototype

This preserves frozen de53 and PVS/history v2 sources. It changes only ROOT breadth: all legal roots receive a CHARGED static/terminal fallback, depth1 searches all roots, depth2 the top8, depth3 top4 and depths4..8 top2, clipped to legal count. Ranking uses the previous COMPLETED pass's fail-soft estimates; ties keep its incumbent first, then deterministic UCI ordering. Beams shrink monotonically: excluded moves do not re-enter. Estimates can be bounds, not exact root action values. An incomplete deeper pass is discarded for move/value selection; all its nodes, evaluator calls, TT/check extensions remain charged and archived.

`root_actions` is ALWAYS full legal fallback coverage. `completed_depth` means completed DECLARED root subset. `full_legal_completed_depth`, per-pass ordered root coverage, float.hex estimates and attempted/incomplete coverage explicitly distinguish the two. PV contains only the selected legal root move; no unsupported multi-ply/depth certification. `root_width="full"` is a control: every original v2 result field (including value HEX), ordered evaluator full-history aliases and counters match on the fixture suite.

Full rule history, root-relative mate magnitude, conservative full-history TT keys, PVS bounds, finite one-charged-incheck evasion extension, qdepth2, maxdepth8, claimed draws and global node accounting are inherited byte-for-byte in their numerical methods from v2. Root node budget is512 for development,8192 only if ROOT later prospectively selects it for own-learning. No budget reset or uncharged alternative exploration. A budget too small to cover all static root actions fails closed.

Beam pruning can discard the actual best tactical/root move. A regression fixture deliberately demonstrates exactly that: full depth2 finds h2h4 at1.0 while the beam prunes it and returns0.0. Depth is therefore not equivalent to full minimax depth. These are conventional selective/beam search and alpha-beta techniques, not a novelty claim. Any baseline gain is SEARCH-ONLY, not self-learning.

## ROOT-only actual development recipe

No model, human-prior forward, SF, actual search or games were executed by this agent. Eight tests use deterministic fake evaluations/pure metadata only. An independent exhaustive depth2 minimax comparator covers castling, EP, promotion, check, repetition,50-move claims, insufficient material, mate/stalemate. Six search tests pass plus two metadata guards, Ruff passes and both actual CLI imports/help pass.

`qualification/register.py` emits a prospective source-bound registration; `qualification/develop.py` inherits the previously executed paired24 / known8×2colors×2engines=32-game HUMAN PRIOR protocol, retaining all ordered full-history evaluator traces,512-node own/SF actual counters and complete UCI histories. New registration/result schemas are `progressive-width-human-prior-development-...-v1`. CPU affinity is the registered CPU (e.g0), not the old hardcoded2. CORE clean6fcc, originalhuman prior, de53 and officialSF19 binary inputs must match original bindings. All output goes to NEW RAM directories, guards remain15GiB/256MiB disk floor/256MiB RAM output. ROOT supplies first/end/operator; bank/model/train sources are never touched.

```
PY qualification/register.py --mode profile --cpu-core 0 --first ROOT_FIRST --deadline ROOT_FIRST_PLUS_AT_MOST_600 --operator-end-epoch ROOT_OPERATOR --core-repo CLEAN6FCC --core-commit 6fcc8b476d25495d1c9c413e55b2c7ba4794013e --old-search DE53 --advanced-search STAGE/search.py --prior-helper HUMAN18 --train-roots FROZEN_TRAIN_POOL --output NEW_RAM_PROFILE --registration NEW_PROFILE_REG
PY qualification/develop.py --registration NEW_PROFILE_REG
```

Fixed first24 frozen TRAIN roots in original order, both methods alternating order, no outcome/score screening. Original median wall ratio<=1.10 remains the fail-fast admission. Optional incheck horizon0 is profile-only, registered BEFORE execution; no unregistered parameter sweep.

Only if actual profile passes, ROOT may stamp a NEW7200s32-game development phase with `--mode arena`, `--profile-result NEW_RAM_PROFILE/result.json --book KNOWN8 --stockfish OFFICIAL_SF19` and the same remaining source/core/CPU bindings. The new helper verifies the actual progressive profile, its helper/source pin, original clock and latency flag before games. UNKNOWN caps remain separate pessimistic and draw-filled diagnostics. Result is development/search-only, not formal strength confirmation.

If baseline evidence warrants a learning experiment, it requires NEW prospective search/native contract/admission source integration with BOTH parent and child on this SAME fixed search. Frozen human-prior-own-v1 and human-randomstarts-own-v2 de53 whitelists remain unchanged. Existing advanced32 actual score.125 versus de53.125 is preserved as a negative ancestor, not relabeled successful or reset.
