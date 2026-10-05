# Scratch-to-source handoff plan (no changes applied)

All files here are a proposal, not part of the active source. If ROOT chooses
to promote this method, copy the new modules as a distinct schema and keep the
classical18 modules/native readers unchanged:

1. Copy `pst_features.py`, `pst_value.py`, `pst_learner.py`, `pst_native.py`,
   `prepare_pst.py`, `mixed_value.py`, and `train_pst.py` into a new
   `experiments/ufuk/cpu-classical-pst-v1/` package. Include `launch_pst.py`
   and `verify_data.py` in the registered helper closure.
2. Copy the three byte-pinned data helpers from `frozen_reference/` and their
   runtime helper, or adapt imports only through a separately reviewed,
   byte-pinned bridge. Do not rename an old classical18 native as a PST native.
3. Freeze the registered fit protocol from the exact two 16,384-action journals,
   sealed actor configs, `data-preflight-receipt-launch-ready-v2.json`, new fit deadlines,
   clean source SHA, and `inventory.json`. Root must update source helper SHA
   closure after copying and repeat both-seed preflight.
4. Freeze both phase clocks and the protocol before any output. The proof phase
   must qualify each seed with whole8/pause4/freshresume8 and strict loads of
   steps 0/4/8 for both seeds. The separate fit phase starts from
   zero 18-term and zero PST residuals with fresh Adam; it does not resume or
   inherit the classical18 fit optimizer.
5. Keep all gates and the later search/arena controls unchanged. A changed
   feature family or new regularizer is not pure capacity control. Preserve
   every failure and require independent full-history arena audit.

ROOT owns any later promotion, clock registration, source integration,
qualification, training, and arena execution.
