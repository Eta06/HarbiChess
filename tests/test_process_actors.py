from pathlib import Path

import pytest

from harbichess.replay.shard import read_shard
from harbichess.training.torch_loop import LoopConfig, run_loop


def test_spawned_real_games_training_resume_and_cross_mode_rejection(tmp_path):
    config = LoopConfig(games=2, simulations=2, max_plies=4, steps=1, batch_size=2, workers=2)
    run = tmp_path / "run"
    first = run_loop(run, config=config, generations=1, actor_mode="process", wall_seconds=60)
    checkpoint = run / first["last_complete_checkpoint"]
    assert first["status"] == "completed"
    assert first["config"]["actor_mode"] == "process"
    assert first["state"]["history"][0]["inference_statistics"]["largest_batch"] == 1
    second = run_loop(
        run, config=config, generations=2, actor_mode="process", resume=checkpoint, wall_seconds=60
    )
    assert second["state"]["generation"] == 2
    assert second["children_cpu_seconds"] > 0
    for shard in (run / "replay").glob("*.gz"):
        assert read_shard(shard).records
    with pytest.raises(ValueError, match="configuration mismatch"):
        run_loop(
            run,
            config=config,
            generations=3,
            actor_mode="thread",
            resume=run / Path(second["last_complete_checkpoint"]),
        )
