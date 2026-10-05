"""ROOT controller support. Fixed local originals, opaque dispatch JSON in RAM only."""

from pathlib import Path

import receiver_v6
import state_bundle as bundle

E8 = Path(
    "/workspace/work/harbichess/a100/restoration/local-rehearsal-content/"
    "harbichess-inputs/initial-e8.safetensors"
)


def geometry():
    paths = {}
    for tag in bundle.TAGS:
        arm, seed = tag.rsplit("-", 1)
        root = (
            Path("/dev/shm")
            / (
                "harbichess-fresh-fullcritic-production-fits-v1-20261005"
                if arm == "fullcritic"
                else "harbichess-fresh-production-fits-v1-20261005"
            )
            / tag
        )
        step = 105 if seed == "20262805" else 97
        paths[f"{tag}/candidate.safetensors"] = root / "candidate.safetensors"
        for name in ("training.pt", "checkpoint.json"):
            paths[f"{tag}/{name}"] = root / "checkpoints" / f"step-{step:08d}" / name
    for seed in (20262805, 20262806):
        paths[f"data/{seed}/actions-00008192.json.gz"] = (
            Path(f"/dev/shm/harbichess-fresh-E0-{seed}") / "actions-00008192.json.gz"
        )
        paths[f"data/{seed}/actor-config.json"] = (
            Path("/workspace/work/harbichess/cpu-fresh-selfplay-v2-actual/registration")
            / f"{seed}-E0-actor-config.json"
        )
    return paths


def reconstruct_local_capsule(manifest):
    receiver_v6.validate_manifest(manifest)
    paths = geometry()
    bundle.require(
        manifest["original_geometry"] == {k: str(v) for k, v in paths.items()},
        "fixed-local-source-geometry",
    )
    originals = {key: path.read_bytes() for key, path in paths.items()}
    bundle.require(bundle.file_rows(originals) == manifest["files"], "local-all22-sealed-bytes")
    capsule, measured = bundle.encode(originals, E8.read_bytes())
    del originals
    bundle.require(
        measured["capsule_sha256"] == manifest["capsule_sha256"]
        and measured["capsule_bytes"] == manifest["capsule_bytes"]
        and measured["chunks"] == manifest["chunks"],
        "fixed-capsule-chunks",
    )
    return capsule
