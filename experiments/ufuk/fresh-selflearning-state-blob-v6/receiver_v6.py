"""One reviewed workflow: fetch exact temporary blob, decode22, publish one Release capsule."""

import json
import os
from pathlib import Path

import blob_delivery_v6 as delivery
import release_transport_v6 as release
import state_bundle as bundle

ROOT = Path(__file__).resolve().parent
DIRECTORY = "experiments/ufuk/fresh-selflearning-state-blob-v6"
WORKFLOW = ".github/workflows/fresh-selflearning-state-blob-v6.yml"
SUPPORT = (
    "native_delta_codec.py",
    "fresh_binding.py",
    "state_bundle.py",
    "release_transport_v6.py",
    "blob_delivery_v6.py",
)


def capsule_name(manifest):
    return f"freshstatev6-sha256-{manifest['capsule_sha256']}.bundle.zlib"


def validate_manifest(manifest):
    bundle.validate(manifest)
    delivery.validate_blob_binding(manifest)
    bundle.require(manifest["status"] == "ROOT-approved-frozen-transport", "manifest-pending")
    bundle.require(
        manifest["workflow_path"] == WORKFLOW
        and manifest["source_commit"] == "6fcc8b476d25495d1c9c413e55b2c7ba4794013e"
        and manifest["action_pins"] == {"checkout": "11bd71901bbe5b1630ceea73d27597364c9af683"},
        "fixed-workflow-source-action",
    )
    bundle.require(
        set(manifest["source_bindings"]) == set(SUPPORT)
        and all(
            bundle.valid_sha(s)
            for s in [
                manifest["workflow_sha256"],
                manifest["receiver_sha256"],
                *manifest["source_bindings"].values(),
            ]
        ),
        "fixed-helper-sha-bindings",
    )


def load_approved():
    digest = os.environ["MANIFEST_SHA"]
    bundle.require(
        bundle.valid_sha(digest) and os.environ["MANIFEST_PREFIX"] == digest[:12],
        "owned-run-manifest-prefix",
    )
    allow = json.loads((ROOT / "allowlist-PENDING.json").read_bytes())
    bundle.require(allow.get(digest) == "manifest-DRAFT.json", "fixed-approved-manifest-allowlist")
    raw = (ROOT / "manifest-DRAFT.json").read_bytes()
    bundle.require(bundle.sha(raw) == digest, "manifest-source-sha")
    manifest = json.loads(raw)
    validate_manifest(manifest)
    bundle.require(
        bundle.sha(Path(__file__).read_bytes()) == manifest["receiver_sha256"], "receiver-source"
    )
    for name, digest in manifest["source_bindings"].items():
        bundle.require(bundle.sha((ROOT / name).read_bytes()) == digest, "helper-source")
    bundle.require(
        bundle.sha((ROOT.parents[2] / WORKFLOW).read_bytes()) == manifest["workflow_sha256"],
        "checked-out-workflow-sha",
    )
    control_raw = (ROOT / "transport-control-PENDING.json").read_bytes()
    control = json.loads(control_raw)
    approval = json.loads((ROOT / "ROOT-approval-PENDING.json").read_bytes())
    bundle.require(
        approval.get("schema") == "fresh-selflearning-state-blob-ROOT-approval-v6"
        and approval.get("status") == "ROOT-approved-frozen-transport"
        and approval.get("manifest_sha256") == os.environ["MANIFEST_SHA"]
        and approval.get("control_sha256") == bundle.sha(control_raw)
        and approval.get("release") == manifest["release"] == control.get("release")
        and control.get("manifest_sha256") == os.environ["MANIFEST_SHA"]
        and approval.get("blob_sha1") == manifest["blob"]["sha1"],
        "ROOT-new-scope-control-release-blob-seal",
    )
    release.check_clock(control)
    return manifest, control


def publish_capsule(manifest, transport, parent, capsule):
    # Verify all22 originals before a permanent asset is written.
    restored = bundle.decode(capsule, parent, manifest)
    del restored
    proof = release.publish_verified(transport, capsule_name(manifest), capsule)
    # A second anonymous full-body read and full22 reconstruction precede PASS.
    readback = transport.download(proof["name"], manifest["capsule_bytes"])
    restored = bundle.decode(readback, parent, manifest)
    proof.update(
        all22_original_byte_sha_verified=True,
        originals=bundle.file_rows(restored),
        scope="exact-byte-public-backup; no independent native runtime or strength proof",
        git_blob_sha1=manifest["blob"]["sha1"],
        release_id=manifest["release"]["id"],
    )
    release.check_clock(transport.control)
    return proof


def main():
    manifest, control = load_approved()
    transport = release.Transport(os.environ["GH_TOKEN"], control, manifest["release"])
    parent = transport.parent()
    capsule = delivery.fetch_verified_blob(transport, manifest)
    proof = publish_capsule(manifest, transport, parent, capsule)
    proof["manifest_sha256"] = os.environ["MANIFEST_SHA"]
    print(json.dumps(proof, sort_keys=True))


if __name__ == "__main__":
    main()
