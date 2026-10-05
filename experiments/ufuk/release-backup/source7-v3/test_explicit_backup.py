import copy
import json
from pathlib import Path

import pack_source7_explicit as pack
import pytest

FIXTURE = Path(__file__).with_name("closed-inventory-compact.json")


def test_actual_eight_native_payload_inventories_are_closed_and_source_pinned():
    obj = json.loads(FIXTURE.read_text())
    pack.validate_inventory(obj)
    assert len(obj["files"]) == 129
    assert len(obj["natives"]) == 8
    result = next(
        x
        for x in obj["files"]
        if x["path"]
        == ("/content/harbichess-runs/certificate-source7-fullshape900-20261005/result.json")
    )
    assert result["sha256"] == "7282af0b37f2da04d2686e52ef4aef10287ec1562a37db5bc5252ba853a196a2"


@pytest.mark.parametrize(
    "mutation", ["source", "open", "duplicate", "outside", "payload", "schema"]
)
def test_inventory_mutations_fail_closed(mutation):
    obj = copy.deepcopy(json.loads(FIXTURE.read_text()))
    if mutation == "source":
        obj["source_commit"] = "0" * 40
    elif mutation == "open":
        obj["natives"][0]["state"]["pending_search_schedule"] = "pending"
    elif mutation == "duplicate":
        obj["files"].append(obj["files"][0])
    elif mutation == "outside":
        obj["files"][0]["path"] = "/content/private/auth.json"
    elif mutation == "payload":
        obj["natives"][0]["artifacts"]["training.pt"] = "0" * 64
    elif mutation == "schema":
        obj["natives"][0]["schema"] = "torch-search-acting-native-cpu-v2"
    with pytest.raises(ValueError):
        pack.validate_inventory(obj)


def test_immutable_input_rejects_symlink_or_changed_bytes(tmp_path):
    target = tmp_path / "source"
    target.write_bytes(b"original")
    digest = pack.sha(target)
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    with pytest.raises(ValueError):
        pack.read_verified(alias, digest)
    target.write_bytes(b"changed")
    with pytest.raises(ValueError):
        pack.read_verified(target, digest)


def test_deadline_expired_rejected_without_training_or_clock_reset():
    with pytest.raises(RuntimeError, match="deadline-expired"):
        pack.guard(0)


@pytest.mark.parametrize(
    "suffix", ["/book.json", "/initial-e8.safetensors", "/tiny-config.json", "/protocol.json"]
)
def test_missing_original_four_input_companion_fails_before_pack(suffix):
    obj = copy.deepcopy(json.loads(FIXTURE.read_text()))
    obj["files"] = [row for row in obj["files"] if not row["path"].endswith(suffix)]
    with pytest.raises(ValueError, match="input-companion"):
        pack.validate_inventory(obj)


@pytest.mark.parametrize(
    "path", ["/absolute", "../escape", "payload/../../escape", "payload\\escape"]
)
def test_restoration_rejects_unsafe_archive_map_paths(path):
    import restore_source7_cuda_proof as restore

    with pytest.raises(ValueError, match="unsafe"):
        restore.safe_relative(path)


def test_restoration_input_geometry_is_exact_without_manifest_rewrite():
    import os

    import restore_source7_cuda_proof as restore

    obj = json.loads(FIXTURE.read_text())
    native = next(
        n
        for n in obj["natives"]
        if n["path"].endswith("/whole/checkpoints/epoch-00000002/checkpoint.json")
    )
    root = Path("/different-content-proof-root")
    parent = root / restore.NATIVE
    for item in native["inputs"].values():
        resolved = Path(os.path.normpath(str(parent / item["relative_path"])))
        assert resolved.is_relative_to(root / "content")
        assert str(resolved).removeprefix(str(root)) in {x["path"] for x in obj["files"]}


def test_exact_prior127_manifest_preserved_and_new128_caps_enforced():
    import extend_manifest as extend

    prior = Path(
        "/workspace/HarbiChess/docs/runs/UFUK-A100-formal45-release-transport-manifest-v2-20261005.json"
    ).read_bytes()
    original = json.loads(prior)
    result = {
        "status": "closed-source7-explicit-archive-pack-pass",
        "source_commit": pack.SOURCE,
        "sha256": "a" * 64,
        "bytes": 127167165,
    }
    out = extend.extend(prior, result, "https://test-reviewed.trycloudflare.com")
    assert len(out["assets"]) == 128
    assert out["assets"][:-1] == original["assets"]
    assert sum(row["bytes"] for row in out["assets"]) == 4492050029
    with pytest.raises(ValueError, match="original127"):
        extend.extend(prior + b" ", result, "https://test-reviewed.trycloudflare.com")
    with pytest.raises(ValueError, match="quicktunnel"):
        extend.extend(prior, result, "https://evil.example")
    result["bytes"] = 1024**3 + 1
    with pytest.raises(ValueError, match="size"):
        extend.extend(prior, result, "https://test-reviewed.trycloudflare.com")
