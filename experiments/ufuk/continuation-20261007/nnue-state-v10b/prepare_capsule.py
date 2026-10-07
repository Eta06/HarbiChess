"""Encode and verify only the frozen Oct7 native/replay allowlist, in RAM."""
import argparse
import json
import os
import sys
from pathlib import Path

import raw_capsule_v10b as codec
import release_transport_v10b as release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_bytes())
    clock = json.loads(args.clock.read_bytes())
    release.check_clock(clock)
    sys.path.insert(0, "/workspace/work/harbichess/cpu-additive-source-6fcc8b4/src")
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)

    def guard():
        release.check_clock(clock)
        memory.check()

    args.output.mkdir(parents=True, exist_ok=False)
    raw = codec.encode(manifest, guard)
    guard()
    originals = codec.verify(raw, manifest)
    guard()
    with (args.output / "capsule.zlib").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    parts = []
    for index, offset in enumerate(range(0, len(raw), 8 * 2**20)):
        part = raw[offset:offset + 8 * 2**20]
        path = args.output / f"part{index:02}.bin"
        with path.open("xb") as stream:
            stream.write(part)
        import hashlib
        digest = codec.sha(part)
        parts.append(dict(index=index, offset=offset, bytes=len(part), sha256=digest,
            git_blob_sha1=hashlib.sha1(b"blob " + str(len(part)).encode() + b"\0" + part).hexdigest(),
            asset_name=f"nnue-state-v10b-{codec.sha(raw)[:12]}-part{index:02}-{digest[:12]}.bin"))
    result = dict(status="PASS-local-allowlisted-capsule-all-original-bytes-not-durable",
        manifest_sha256=codec.sha(args.manifest.read_bytes()),
        full_capsule_bytes=len(raw), full_capsule_sha256=codec.sha(raw),
        original_files=len(originals), raw_bytes=manifest["raw_bytes"], parts=parts,
        strength_success=False, runtime_restore_claimed=False)
    with (args.output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != "parts"}), flush=True)


if __name__ == "__main__":
    main()
