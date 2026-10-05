"""Review-only V7 receiver bridge; ROOT must freeze exact code/control before execution."""

import base64
import hashlib
import json
import re
from pathlib import Path

import raw_capsule_v7 as capsule

REPO = "Eta06/HarbiChess"
API = "https://api.github.com"
RELEASE = 404068972
TAG = "ufuk-cpu-ownplay-state-20261005"
PROTECTED = (
    "freshstatev6-sha256-"
    "66ef6b4f05a9ff3d404debb424d477860b2a7439c7ce51fe91c4c6d3399409ff.bundle.zlib"
)


def git_sha(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def validate_control(control, manifest_raw):
    if (
        control["schema"] != "classical-raw-blob-release-control-v7"
        or control["approval"] != "ROOT-approved"
    ):
        raise ValueError("new approval required")
    if control["manifest_sha256"] != capsule.sha(manifest_raw):
        raise ValueError("manifest SHA")
    if control["release_id"] != RELEASE or control["release_tag"] != TAG:
        raise ValueError("fixed existing authorized release")
    if PROTECTED not in control["protected_assets"]:
        raise ValueError("old V6 asset must be preserved")
    if (
        control["asset_name"]
        != "classical-state-v7-sha256-" + control["capsule_sha256"] + ".bundle.zlib"
    ):
        raise ValueError("new unique content-addressed asset")
    if not 0 < control["capsule_bytes"] <= capsule.ENCODED_LIMIT:
        raise ValueError("encoded bound")
    if not re.fullmatch("[0-9a-f]{40}", control["blob_sha1"]):
        raise ValueError("blob id")
    if control["deadline"] > 1791273600 or not 0 < control["deadline"] - control["first"] <= 5400:
        raise ValueError("new scoped clock/operator cap")
    if control.get("source_commit") != "6fcc8b476d25495d1c9c413e55b2c7ba4794013e":
        raise ValueError("original core source binding")
    helper_names = {"raw_capsule_v7.py", "receiver_v7.py", "release_transport_v7.py"}
    if set(control.get("helper_sha256", {})) != helper_names:
        raise ValueError("exact helper source closure")
    for name in helper_names:
        if (
            capsule.sha(Path(__file__).with_name(name).read_bytes())
            != control["helper_sha256"][name]
        ):
            raise ValueError("helper source SHA")
    if not re.fullmatch("[0-9a-f]{64}", control.get("workflow_sha256", "")):
        raise ValueError("workflow seal missing")
    clock = control.get("clock", {})
    if clock != {
        "schema": "classical-raw-blob-release-clock-v7",
        "status": "ROOT-approved-frozen-transport",
        "operator_end_epoch": 1791273600,
        "started_epoch": control["first"],
        "deadline_epoch": control["deadline"],
    }:
        raise ValueError("shared new-scope clock binding")
    manifest = json.loads(manifest_raw)
    capsule.checked_rows(manifest)
    return manifest


def decode_blob(raw_json, control):
    n = 4 * ((control["capsule_bytes"] + 2) // 3)
    if len(raw_json) > n * 11 // 10 + 16384:
        raise ValueError("blob JSON bound")
    value = json.loads(raw_json)
    url = API + f"/repos/{REPO}/git/blobs/" + control["blob_sha1"]
    if (value.get("sha"), value.get("size"), value.get("encoding"), value.get("url")) != (
        control["blob_sha1"],
        control["capsule_bytes"],
        "base64",
        url,
    ):
        raise ValueError("fixed repo/id/size/encoding")
    content = value.get("content")
    if not isinstance(content, str) or re.fullmatch("[A-Za-z0-9+/=\r\n]*", content) is None:
        raise ValueError("base64 alphabet")
    clean = content.replace("\r", "").replace("\n", "")
    if len(clean) != n:
        raise ValueError("base64 length")
    raw = base64.b64decode(clean, validate=True)
    if (
        len(raw) != control["capsule_bytes"]
        or capsule.sha(raw) != control["capsule_sha256"]
        or git_sha(raw) != control["blob_sha1"]
    ):
        raise ValueError("full SHA256/GitSHA1")
    return raw


def publish_verified(control, manifest_raw, blob_json, transport):
    """Injected reviewed V6 transport adapter; this module itself makes no requests."""
    manifest = validate_control(control, manifest_raw)
    raw = decode_blob(blob_json, control)
    proofs = capsule.verify(raw, manifest)
    # Adapter must retain canonical GET retries, no ambiguous POST retry, 200/8GiB,
    # immutable matching-asset idempotency and FULL anonymous readback from V6.
    import release_transport_v7 as release

    public = release.publish_verified(transport, control["asset_name"], raw)
    if (
        public["sha256"] != control["capsule_sha256"]
        or public["bytes"] != len(raw)
        or not public["anonymous_full_body_sha_verified"]
    ):
        raise ValueError("public full-byte proof required")
    return {
        "schema": "classical-raw-public-preservation-proof-v7",
        "status": "PASS-byte-preservation-only",
        "raw_files": proofs,
        "public": public,
        "runtime_restore_claimed": False,
    }


def execute_approved(control, manifest_raw, token):
    """Future ROOT workflow entrypoint; never called by this preparation task."""
    import release_transport_v7 as release

    validate_control(control, manifest_raw)
    binding = {
        "id": RELEASE,
        "tag": TAG,
        "target_commitish": "31a18e981e27e696fcca69c7e62210e7f0e89830",
        "initial_assets": 1,
    }
    transport = release.Transport(token, control["clock"], binding)
    transport.inventory()  # Existing V6 asset + fixed release identity + 200/8GiB.
    n = 4 * ((control["capsule_bytes"] + 2) // 3)
    url = API + f"/repos/{REPO}/git/blobs/" + control["blob_sha1"]
    with transport.request(url) as response:
        body = response.read(n * 11 // 10 + 16384 + 1)
    return publish_verified(control, manifest_raw, body, transport)
