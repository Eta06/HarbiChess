# Fresh own-learning arena: MC, SC, and full value critic

Prospective CPU-only development arena for the six completed fresh own-data fits. This stage uses the common fixed E8 baseline and three fitted candidates per seed: own terminal Monte Carlo (MC), own-search consistency (SC), and full value critic (FULL). It evaluates all candidates at the same Q512/qdepth-2 search configuration.

No fits, actor generation, games, or strength selection happen in this directory. `build_fit_manifest.py` reads already-completed fit/native receipts and produces `fit-provenance.json`. The manifest binds each candidate file to the fit cohort row, result JSON, fit contract, initial step-0 native checkpoint, and final native checkpoint. It deliberately does not look for a checkpoint next to the candidate file.

## Prospective hypotheses and controls

- Hypothesis: at least one fresh own-experience critic improves over the unchanged E8 model under the same inference search, and reaches the frozen SF512 screen.
- Controls: E8 vs SF512 is run for both seeds; each MC/SC/FULL model is paired against E8 and SF512 on the same eight opening pairs and both candidate colors.
- Data/source: source `6fcc8b476d25495d1c9c413e55b2c7ba4794013e`; six completed fit receipts and the already frozen eight-pair development book. No fit checkpoint or arm is selected after these results.
- Budget: exactly 224 games, seven tournaments per seed at 16 games each; 512 neural nodes, qdepth 2, max depth 8, max 400 total plies; Stockfish requested 512 nodes, one thread, Hash 16 MiB. Profile budget is 24 fixed neural searches, three learned roles × two seeds × four book histories.
- Frozen thresholds are evaluated for each learned role on both seeds: candidate score vs E8 strictly above 0.60; paired SF512 gain over E8 strictly above 0.10; SF512 score at least 0.25; caps at most 0.05. A role is development-eligible only if every gate passes on both seeds. Other roles may fail without disqualifying an eligible role. This remains development evidence, not final virgin confirmation.
- Stop: stop at the prospectively fixed whole-stage deadline; preserve failed/incomplete rows. Do not change the sample, threshold, candidate, or search after reading match results.

`protocol-TEMPLATE.json` is not registered until the owner freezes actual clocks and final hashes. `profile.py` checks 24 fixed starts and search budgets, then writes a publish-once qualification receipt. `run.py` checks that receipt and all portable fit/native bindings before running the fixed cohort. `audit_arena.py` independently replays every game and computes paired gates without running another engine or model search.

The profile and arena must execute against a clean checkout of the pinned source. An external owner/guardian is responsible for the original clocks and whole-process lifecycle. No model promotion follows from an arena screen alone.
