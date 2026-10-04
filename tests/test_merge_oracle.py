import copy
import json

import pytest
from test_prepared_oracle import make_dataset

from harbichess.training.merge_oracle import merge
from harbichess.training.oracle_data import read_game
from harbichess.training.prepared_oracle import PreparedOracle, digest, prepare


def test_merge_preserves_anchor_bytes_targets_and_all_heldout_keys(tmp_path):
    anchor, broader, result = (tmp_path / name for name in ("anchor", "broader", "merged"))
    make_dataset(anchor)
    make_dataset(broader)
    original = {str(p): p.read_bytes() for source in (anchor, broader) for p in source.iterdir()}
    manifest = merge(result, anchor, broader)
    assert manifest["status"] == "completed" and manifest["derivation_schema"] == 1
    assert manifest["counts"] == {"anchor": 6, "broader": 5, "dropped_broader_training_overlap": 1}
    assert manifest["rows"] == 11 and manifest["games"] == 4
    assert all(
        p.read_bytes() == original[str(p)] for source in (anchor, broader) for p in source.iterdir()
    )
    for name in ("train.json.gz", "validation.json.gz"):
        assert (result / ("anchor-" + name)).read_bytes() == (anchor / name).read_bytes()
        parent = read_game(broader / name)
        derived = read_game(result / ("broader-" + name))
        split = parent["job"]["split"]
        offset = 100000 if split == "train" else 200000
        assert derived["job"]["family"] == parent["job"]["family"] + offset
        by_key = {row["position_key"]: row for row in parent["rows"]}
        for row in derived["rows"]:
            restored = copy.deepcopy(row)
            restored["family"] = restored.pop("source_family")
            assert restored == by_key[row["position_key"]]
    heldout = {
        row["position_key"]
        for source in (anchor, broader)
        for row in read_game(source / "validation.json.gz")["rows"]
    }
    assert all(
        row["position_key"] not in heldout
        for row in read_game(result / "broader-train.json.gz")["rows"]
    )
    # Unchanged anchor train overlaps a broader holdout in this adversarial fixture;
    # the existing reader removes those validation rows rather than altering anchor.
    cache = tmp_path / "prepared"
    prepare(cache, source=result)
    train, validation, info = PreparedOracle(cache, source=result).panels()
    assert info["removed_validation_position_overlap"] == 2
    assert train.size == 5 and validation.size == 4
    with pytest.raises(FileExistsError):
        merge(result, anchor, broader)


@pytest.mark.parametrize("fault", ["incomplete", "source-hash", "row-count", "heldout-key"])
def test_invalid_parent_cannot_publish_a_merged_dataset(tmp_path, fault):
    anchor, broader, output = (tmp_path / name for name in ("anchor", "broader", "refused"))
    make_dataset(anchor)
    make_dataset(broader)
    path = broader / "dataset.json"
    manifest = json.loads(path.read_text())
    if fault == "incomplete":
        manifest["status"] = "incomplete"
    elif fault == "source-hash":
        manifest["files"]["train.json.gz"] = "0" * 64
    elif fault == "row-count":
        manifest["rows"] += 1
    else:
        import gzip

        game_path = broader / "validation.json.gz"
        game = read_game(game_path)
        game["rows"][0]["position_key"] = "false heldout provenance"
        game_path.write_bytes(gzip.compress(json.dumps(game).encode(), mtime=0))
        manifest["files"][game_path.name] = digest(game_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        merge(output, anchor, broader)
    assert not (output / "dataset.json").exists()
