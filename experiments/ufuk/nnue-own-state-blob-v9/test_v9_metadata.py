"""Pure metadata controls only; no capsule build/decode, native/runtime/network."""

import copy

import pytest
import raw_capsule_v9 as c


def manifest(n=1):
    rows = [
        dict(
            path=f"/synthetic/{i}",
            member=f"files/{i:04d}",
            role="fixture",
            bytes=1,
            sha256="0" * 64,
        )
        for i in range(n)
    ]
    return dict(
        schema=c.SCHEMA,
        approval="ROOT-approved-exact-files",
        rows=rows,
        row_count=n,
        raw_bytes=n,
        limits={
            "files": 2048,
            "raw_bytes": c.RAW_LIMIT,
            "encoded_bytes": c.ENCODED_LIMIT,
            "file_bytes": c.FILE_LIMIT,
            "header_bytes": 2**20,
        },
    )


def test_new_prospective_cap_supports_full1054_not_v7_caps():
    assert len(c.checked_rows(manifest(1054))) == 1054
    with pytest.raises(ValueError):
        c.checked_rows(manifest(2049))


def test_requires_exact_large_cap_approval_and_measurement():
    x = manifest()
    x["limits"]["raw_bytes"] = 16 * 2**20
    with pytest.raises(ValueError):
        c.checked_rows(x)
    x = manifest()
    x["raw_bytes"] = 2
    with pytest.raises(ValueError):
        c.checked_rows(x)


def test_alias_size_and_pending_manifest_fail_closed():
    x = manifest(2)
    x["rows"][1]["member"] = x["rows"][0]["member"]
    with pytest.raises(ValueError):
        c.checked_rows(x)
    x = manifest()
    x["rows"][0]["bytes"] = c.FILE_LIMIT + 1
    x["raw_bytes"] = x["rows"][0]["bytes"]
    with pytest.raises(ValueError):
        c.checked_rows(x)


def test_old_caps_require_a_distinct_version():
    x = manifest()
    x["limits"].update(raw_bytes=128 * 2**20, encoded_bytes=32 * 2**20, file_bytes=2 * 2**20)
    with pytest.raises(ValueError):
        c.checked_rows(x)
    x = manifest()
    x["approval"] = "PENDING"
    with pytest.raises(ValueError):
        c.checked_rows(x)


def test_role_path_full_original_constraints():
    x = manifest()
    bad = copy.deepcopy(x)
    bad["rows"][0]["path"] = "relative"
    with pytest.raises(ValueError):
        c.checked_rows(bad)
    x["rows"][0]["sha256"] = "not-actual-sha"
    with pytest.raises(ValueError):
        c.checked_rows(x)


def test_quiescence_exact_scope_clock_and_no_remaining(tmp_path):
    import json

    from prepare_v9 import check_quiescence, sha

    cfg = dict(
        regular_file_roots=["/fixed/root"],
        exact_files=["/fixed/file"],
        source_commit="source",
        clock=dict(first=100, deadline=5500),
    )
    receipt = tmp_path / "receipt.json"
    good = dict(
        status="PASS-closed-V9-scope-owned-quiescence",
        remaining_owned=[],
        scope_roots=cfg["regular_file_roots"],
        exact_files=cfg["exact_files"],
        source_commit="source",
        clock=cfg["clock"],
        observed_epoch=101,
    )

    def bind(value):
        receipt.write_text(json.dumps(value))
        cfg["scope_quiescence_receipt"] = dict(path=str(receipt), sha256=sha(receipt))

    bind(good)
    check_quiescence(cfg, 102)
    for field, value in [
        ("remaining_owned", [123]),
        ("scope_roots", []),
        ("exact_files", []),
        ("source_commit", "other"),
        ("observed_epoch", 103),
        ("clock", dict(first=101, deadline=5501)),
    ]:
        bad = copy.deepcopy(good)
        bad[field] = value
        bind(bad)
        with pytest.raises(ValueError):
            check_quiescence(cfg, 102)
    bind(good)
    receipt.write_text("{}")
    with pytest.raises(ValueError):
        check_quiescence(cfg, 102)


def test_explicit_scope_detects_nested_added_tail(tmp_path):
    from prepare_v9 import files

    root = tmp_path / "root"
    root.mkdir()
    (root / "native0").write_text("original")
    cfg = dict(regular_file_roots=[str(root)], exact_files=[])
    before = files(cfg)
    nested = root / "checkpoint"
    nested.mkdir()
    (nested / "rng.json").write_text("new state")
    assert len(files(cfg)) == 2
    assert files(cfg) != before
    (root / "alias").symlink_to(root / "native0")
    with pytest.raises(ValueError):
        files(cfg)
