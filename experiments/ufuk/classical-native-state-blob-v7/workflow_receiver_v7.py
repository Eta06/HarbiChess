"""Fixed V7 approved files; no network on import or local validation."""

import json
import os
from pathlib import Path

import raw_capsule_v7 as capsule
import receiver_v7 as receiver
import release_transport_v7 as release

ROOT = Path(__file__).resolve().parent
DIRECTORY = "experiments/ufuk/classical-native-state-blob-v7"
WORKFLOW = ".github/workflows/classical-native-state-blob-v7.yml"
SUPPORT = {"raw_capsule_v7.py", "receiver_v7.py", "release_transport_v7.py"}


def require(value, code):
    if not value:
        raise ValueError(code)


def load_approved(digest=None, prefix=None, root=ROOT, workflow=None):
    digest = os.environ["MANIFEST_SHA"] if digest is None else digest
    prefix = os.environ["MANIFEST_PREFIX"] if prefix is None else prefix
    require(
        receiver.re.fullmatch("[0-9a-f]{64}", digest) is not None and prefix == digest[:12],
        "manifest-prefix",
    )
    allow = json.loads((root / "allowlist.json").read_bytes())
    require(allow == {digest: "manifest.json"}, "one-fixed-approved-manifest")
    manifest_raw = (root / "manifest.json").read_bytes()
    require(capsule.sha(manifest_raw) == digest, "manifest-actual-SHA")
    manifest = json.loads(manifest_raw)
    capsule.checked_rows(manifest)
    require(
        len(manifest["rows"]) == 253 and sum(r["bytes"] for r in manifest["rows"]) == 9208873,
        "fixed-253-original-scope",
    )
    require(
        manifest["workflow_path"] == WORKFLOW
        and manifest["source_commit"] == "6fcc8b476d25495d1c9c413e55b2c7ba4794013e",
        "source-workflow-binding",
    )
    require(
        manifest["action_pins"] == {"checkout": "11bd71901bbe5b1630ceea73d27597364c9af683"},
        "fixed-checkout-pin",
    )
    require(
        set(manifest["source_bindings"]) == SUPPORT | {"workflow_receiver_v7.py"},
        "exact-four-source-bindings",
    )
    for name, expected in manifest["source_bindings"].items():
        require(capsule.sha((root / name).read_bytes()) == expected, "runtime-helper-SHA")
    workflow = root.parents[2] / WORKFLOW if workflow is None else workflow
    require(
        capsule.sha(workflow.read_bytes()) == manifest["workflow_sha256"], "actual-workflow-SHA"
    )
    control_raw = (root / "control.json").read_bytes()
    control = json.loads(control_raw)
    receiver.validate_control(control, manifest_raw)
    require(control["workflow_sha256"] == manifest["workflow_sha256"], "control-workflow-seal")
    require(
        control["helper_sha256"] == {k: manifest["source_bindings"][k] for k in SUPPORT},
        "control-support-seals",
    )
    approval = json.loads((root / "ROOT-approval.json").read_bytes())
    require(
        approval
        == {
            "schema": "classical-native-state-blob-ROOT-approval-v7",
            "status": "ROOT-approved-frozen-transport",
            "manifest_sha256": digest,
            "control_sha256": capsule.sha(control_raw),
            "blob_sha1": control["blob_sha1"],
            "release_id": receiver.RELEASE,
            "scope": "253-original-byte-public-preservation-not-runtime-strength",
        },
        "ROOT-exact-approval-seal",
    )
    release.check_clock(control["clock"])
    return manifest, control


def main():
    manifest, control = load_approved()
    proof = receiver.execute_approved(
        control, (ROOT / "manifest.json").read_bytes(), os.environ["GH_TOKEN"]
    )
    # Publication already includes full anonymous capsule readback. ROOT dispatcher
    # independently downloads and verifies every original again before local PASS.
    proof["manifest_sha256"] = os.environ["MANIFEST_SHA"]
    print(json.dumps(proof, sort_keys=True))


if __name__ == "__main__":
    main()
