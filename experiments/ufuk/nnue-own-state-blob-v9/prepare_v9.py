"""ROOT-only metadata measurement/freeze; no capsule encode/decode or network."""

import argparse
import hashlib
import json
import time
from pathlib import Path

LIMITS = {
    "files": 2048,
    "raw_bytes": 384 * 1024**2,
    "encoded_bytes": 64 * 1024**2,
    "file_bytes": 8 * 1024**2,
    "header_bytes": 1024 * 1024,
}
SUPPORT = {
    "raw_capsule_v9.py",
    "receiver_v9.py",
    "release_transport_v9.py",
    "workflow_receiver_v9.py",
}
PRODUCERS = {"prepare_v9.py", "build_seal_v9.py"}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def files(config):
    result = set()
    for root in config["regular_file_roots"]:
        p = Path(root)
        if not p.is_dir():
            raise ValueError("required scope root unavailable " + root)
        for f in p.rglob("*"):
            if "__pycache__" in f.parts:
                continue
            if f.is_symlink():
                raise ValueError("symlink requires explicit upstream binding " + str(f))
            if f.is_file():
                result.add(f.resolve())
    for name in config["exact_files"]:
        p = Path(name)
        if p.is_symlink() or not p.is_file():
            raise ValueError("required exact regular file " + name)
        result.add(p.resolve())
    return sorted(result)


def check_closed(config):
    for receipt in config["required_closed_arenas"]:
        p = Path(receipt["path"])
        x = json.loads(p.read_bytes())
        if x["status"] != "completed-games-not-strength" or len(x["rows"]) != receipt["groups"]:
            raise ValueError("arena not closed " + str(p))
        for row in x["rows"]:
            result = p.parent / f"{row['seed']}-{row['arm']}-vs-{row['opponent']}.json"
            if sha(result) != row["result_sha256"]:
                raise ValueError("closed raw game SHA")
            if len(json.loads(result.read_bytes())["games"]) != 16:
                raise ValueError("full group games")


def check_quiescence(config, now):
    q = config["scope_quiescence_receipt"]
    if not q["path"] or not q["sha256"] or sha(q["path"]) != q["sha256"]:
        raise ValueError("ROOT exact ownedquiescence receipt")
    witness = json.loads(Path(q["path"]).read_bytes())
    if (
        witness["status"] != "PASS-closed-V9-scope-owned-quiescence"
        or witness["remaining_owned"]
        or witness["scope_roots"] != sorted(config["regular_file_roots"])
        or witness["exact_files"] != sorted(config["exact_files"])
        or witness["source_commit"] != config["source_commit"]
        or witness["clock"] != config["clock"]
        or not config["clock"]["first"] <= witness["observed_epoch"] <= now
    ):
        raise ValueError("pending/live or differently bound scope")
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    a = parser.parse_args()
    c = json.loads(a.config.read_bytes())
    paths = files(c)
    count = len(paths)
    raw = sum(p.stat().st_size for p in paths)
    maximum = max(p.stat().st_size for p in paths)
    if count > LIMITS["files"] or raw > LIMITS["raw_bytes"] or maximum > LIMITS["file_bytes"]:
        raise ValueError("prospective caps exceeded; do not silently omit states")
    report = dict(
        status="metadata-only-NOT-frozen-NOT-public",
        files=count,
        raw_bytes=raw,
        max_file_bytes=maximum,
        encoded_bytes=None,
        limits=LIMITS,
        rows=[dict(path=str(p), bytes=p.stat().st_size, sha256=None) for p in paths],
        pending_closed_arenas=c["required_closed_arenas"],
    )
    if a.freeze:
        clock = c["clock"]
        now = time.time()
        if (
            c["approval"] != "ROOT-approved-new-V9-scope"
            or c["limits"] != LIMITS
            or not clock["first"]
            <= now
            < clock["deadline"]
            <= min(clock["first"] + 5400, 1791273600)
        ):
            raise ValueError("new observed5400 ROOTscope before freeze/build")
        check_closed(c)
        q = check_quiescence(c, now)
        helper = Path(c["helper_directory"])
        workflow = Path(c["workflow_path"])
        bindings = {name: sha(helper / name) for name in SUPPORT}
        rows = [
            dict(
                path=str(p),
                member=f"files/{i:04d}",
                role="explicit-V9-byte-preservation-NOT-runtime-strength",
                bytes=p.stat().st_size,
                sha256=sha(p),
            )
            for i, p in enumerate(paths)
        ]
        if files(c) != paths or any(
            p.stat().st_size != r["bytes"] for p, r in zip(paths, rows, strict=True)
        ):
            raise ValueError("input changed during freeze")
        report = dict(
            schema="classical-explicit-raw-native-capsule-v9",
            approval="ROOT-approved-exact-files",
            status="frozen-scope-NOT-built-NOT-public",
            rows=rows,
            row_count=count,
            raw_bytes=raw,
            limits=LIMITS,
            source_commit=c["source_commit"],
            release_id=404068972,
            release_tag="ufuk-cpu-ownplay-state-20261005",
            workflow_path=".github/workflows/nnue-own-state-blob-v9.yml",
            workflow_sha256=sha(workflow),
            source_bindings=bindings,
            action_pins={"checkout": "11bd71901bbe5b1630ceea73d27597364c9af683"},
            coverage_cutoff=c["coverage_cutoff"],
            external_dependencies=c["external_dependencies"],
            parent_preservation=c["parent_preservation"],
            new_scope_clock=clock,
            scope_roots=sorted(c["regular_file_roots"]),
            exact_files=sorted(c["exact_files"]),
            scope_config_sha256=sha(a.config),
            scope_quiescence_receipt=q,
            producer_bindings={name: sha(helper / name) for name in PRODUCERS},
        )
        if time.time() >= clock["deadline"]:
            raise TimeoutError("new original5400 exhausted during manifest freeze")
    with a.output.open("x") as f:
        json.dump(report, f, indent=2)
    print(
        json.dumps(
            {k: report[k] for k in ["status", "raw_bytes"]}
            | {"files": count, "encoded_bytes": None}
        )
    )


if __name__ == "__main__":
    main()
