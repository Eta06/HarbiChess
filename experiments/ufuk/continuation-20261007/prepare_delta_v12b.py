"""Freeze only CLOSED teacher-free states, failed attempts and raw evidence; no hot actors."""

import hashlib
import importlib.util
import json
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "nnue-state-v12b"
RAM = Path("/dev/shm")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location(
        "deltaB_codec", OUT / "raw_capsule_v12b.py"
    )
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    selected = {}

    def include(path, role):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        root = path if path.is_dir() else path.parent
        for f in files:
            if (
                f.is_file()
                and not f.is_symlink()
                and not any(
                    x.startswith(".") or x == "__pycache__"
                    for x in f.relative_to(root).parts
                )
            ):
                selected[str(f)] = (role, f.stat().st_size, sha(f))

    for seed in (20262905, 20262906):
        for phase in ("proof", "fresh-fit"):
            result = json.loads(
                (
                    BASE / f"forensic-own-v4-actual-{seed}/{phase}/result.json"
                ).read_bytes()
            )
            expected = (
                "PASS-forensic-v4-whole-pause-fresh-native-proof-not-strength"
                if phase == "proof"
                else "PASS-forensic-v4-fresh64-native-loads-not-strength"
            )
            if result["status"] != expected:
                raise ValueError(
                    "all selected native phases must be actually CLOSED PASS"
                )
        for record in (f"human-random-collect-actual-{seed}",):
            if (
                json.loads((BASE / record / "result.json").read_bytes())["status"]
                != "PASS-actual-procedural-own-Q-collection-not-strength"
            ):
                raise ValueError("raw producer not CLOSED")
    for group in (
        "harbichess-human-prior-zero-v1",
        "harbichess-human-prior-synthetic-v1",
        "harbichess-human-randomstarts-bank-v2",
        "harbichess-human-randomstarts-ownq-v2",
        "harbichess-human-randomstarts-forensic-audit-v3",
        "harbichess-human-randomstarts-forensic-data-v4",
        "harbichess-human-randomstarts-forensic-native-v4",
        "harbichess-TDLeaf-search-equivalence-v4",
        "harbichess-TDLeaf-search-equivalence-v5",
    ):
        include(
            RAM / group, "closed-teacher-free-native-data-fullquery-proof-or-failure"
        )
    for group in ("ranking-known160-data", "MC-known160-data"):
        include(
            RAM / "harbichess-continuation-20261007" / group,
            "closed-negative-known160-games",
        )
    for seed in (20262905, 20262906):
        for prefix in (
            "human-zero-actual",
            "human-random-bank-actual",
            "human-random-collect-actual",
            "procedural-own-audit-actual",
            "procedural-forensic-v3-audit-actual",
            "procedural-forensic-v3-conversion-actual",
            "procedural-forensic-v4-conversion-actual",
            "TDLeaf-v4-equivalence-actual",
            "TDLeaf-v5-equivalence-actual",
            "forensic-own-v4-actual",
        ):
            include(
                BASE / f"{prefix}-{seed}", "closed-ROOT-source-clock-owner-boundary"
            )
        include(
            RAM / "harbichess-continuation-20261007" / f"forensic-v3-{seed}",
            "closed-forensic-view-not-rewritten-receipt",
        )
    for group in (
        "ranking-root-independent-audit",
        "MC-root-independent-audit",
        "human-prior-own-v1",
        "human-randomstarts-own-v2",
        "human-randomstarts-own-v3-forensic-shadow",
        "human-randomstarts-own-v4-forensic-native",
        "human-randomstarts-forensic-manifest-builder-v4",
        "tdleaf-search-equivalence-v4-protected-source-cohort",
        "tdleaf-human-prior-own-v2-qualified-producer-v3",
    ):
        include(BASE / group, "closed-original-or-prospective-source-sha")
    for name in (
        "forensic-v4-inventory.json",
        "procedural-forensic-v4-actual-audit-set.json",
        "procedural-forensic-v3-actual-audit-set.json",
        "procedural-forensic-v3-original-clock-aliases.json",
        "execute_TDLeaf_search_qualification_v5.py",
        "execute_procedural_forensic_audit_v3.py",
        "execute_procedural_forensic_conversion_v3.py",
        "execute_procedural_forensic_conversion_v4.py",
    ):
        include(BASE / name, "closed-ROOT-integration-or-preserved-failure")
    rows = [
        dict(path=p, member=f"files/{i:04}", role=role, bytes=n, sha256=h)
        for i, (p, (role, n, h)) in enumerate(sorted(selected.items()))
    ]
    manifest = dict(
        schema=codec.SCHEMA,
        approval="ROOT-approved-exact-files",
        limits=dict(
            files=2048,
            raw_bytes=codec.RAW_LIMIT,
            encoded_bytes=codec.ENCODED_LIMIT,
            file_bytes=codec.FILE_LIMIT,
            header_bytes=1024 * 1024,
        ),
        rows=rows,
        row_count=len(rows),
        raw_bytes=sum(r["bytes"] for r in rows),
        dependencies=[
            "V12A closed teacher-bootstrap native and negative arenas",
            "V10b exact H0/prior/inference/core ancestor sources",
            "clean core6fcc",
        ],
        hot_TDLeaf_actors_excluded=True,
        strength_success=False,
        scope="CLOSED actual teacher-free own64 native/full Adam/RNG, zero/banks/raw replay and all failed proofs; not strength promotion",
    )
    codec.checked_rows(manifest)
    first = time.time()
    clock = dict(
        schema="NNUE-own-Oct7-deltaB-release-clock-v12b",
        status="ROOT-approved-frozen-transport",
        started_epoch=first,
        deadline_epoch=min(first + 5400, 1791448916.685839),
        operator_end_epoch=1791448916.685839,
    )
    for name, value in [("manifest.json", manifest), ("clock.json", clock)]:
        with (OUT / name).open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    print(
        json.dumps(
            dict(
                files=len(rows),
                raw_bytes=manifest["raw_bytes"],
                max_file=max(r["bytes"] for r in rows),
                first=first,
            )
        )
    )


if __name__ == "__main__":
    main()
