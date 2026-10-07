# Dormant TD(lambda) target transformer

Purely transforms final, per-episode own-Q label rows into fixed-`lambda=0.5` regression targets. It is not connected to a trainer, does not load sealed data, and has not been run on actual collection data.

Input rows must already be final labels from the own-Q v2 packet, grouped by `root_id` and ordered by contiguous `local_ply`. Required fields are `mover`, `clipped_q_mover`, `episode_status`, `own_wdl_mover`, `train_eligible`, `selected_action_played`, and `protected_search_aliases`. No caller should substitute the live `search_row` event stream, which is emitted before episode labels finalize.

`transform_episode(rows)` returns row-aligned `target_white` and `target_mover` values plus source row IDs. It excludes protected episodes wholesale. For a known terminal episode, the last pre-action row gets the exact own W/D/L return and prior rows recursively mix the next logged root Q with the later return. For UNKNOWN endpoints, the last logged pre-action Q is the bootstrap; the transformer does not infer a value for the final board after that action. The input mappings are not mutated.

Synthetic-only verification:

```sh
PYTHONDONTWRITEBYTECODE=1 /workspace/HarbiChess/.venv/bin/pytest -q -p no:cacheprovider test_targets.py
```

This prototype requires a separate source/data registration, sealed-packet digest, predeclared row mask/sample budget, full native-state proof, and root approval before it could be wired into an experiment.
