"""One approved eight-part publication; full capsule/original SHA verification."""

import base64
import hashlib
import json
import os
from pathlib import Path

import raw_capsule_v9 as capsule
import release_transport_v9 as release

HERE = Path(__file__).parent


def validate(c, manifest_raw):
    if (
        c.get("schema") != "NNUE-V9-exact-capsule-eight-parts-v1"
        or c.get("status") != "ROOT-approved-before-part-POST"
        or c.get("original_manifest_sha256") != capsule.sha(manifest_raw)
        or c.get("part_limit_bytes") != 8 * 1024**2
        or c.get("max_parts") != 8
        or len(c.get("parts", [])) != 8
        or c.get("release_id") != 404068972
        or c.get("release_tag") != "ufuk-cpu-ownplay-state-20261005"
    ):
        raise ValueError("exact ROOT-approved scope/manifest/eight-part limits")
    offset, names = 0, set()
    for i, row in enumerate(c["parts"]):
        if (
            row["index"] != i
            or row["offset"] != offset
            or type(row["bytes"]) is not int
            or not 0 < row["bytes"] <= 8 * 1024**2
            or len(row["sha256"]) != 64
            or len(row["git_blob_sha1"]) != 40
            or row["asset_name"] in names
            or row["asset_name"]
            != (
                f"nnue-state-v9-{c['full_capsule_sha256'][:12]}-"
                f"part{i:02d}-{row['sha256'][:12]}.bin"
            )
        ):
            raise ValueError("ordered bounded content-addressed nonoverlapping parts")
        offset += row["bytes"]
        names.add(row["asset_name"])
    if offset != c["full_capsule_bytes"] or offset > capsule.ENCODED_LIMIT:
        raise ValueError("full exact length and original encoded cap")
    release.check_clock(c["clock"])


def git_bytes(body, row):
    data = json.loads(body)
    if (
        data.get("sha") != row["git_blob_sha1"]
        or data.get("size") != row["bytes"]
        or data.get("encoding") != "base64"
    ):
        raise ValueError("canonical Git blob identity/size/encoding")
    raw = base64.b64decode("".join(data["content"].split()), validate=True)
    if (
        len(raw) != row["bytes"]
        or capsule.sha(raw) != row["sha256"]
        or hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        != row["git_blob_sha1"]
    ):
        raise ValueError("full part bytes and both content hashes")
    return raw


def main():
    raw_control = (HERE / "parts-control.json").read_bytes()
    if capsule.sha(raw_control) != os.environ["PARTS_CONTROL_SHA"]:
        raise ValueError("exact registered control SHA")
    c = json.loads(raw_control)
    mraw = (HERE / "manifest.json").read_bytes()
    validate(c, mraw)
    m = json.loads(mraw)
    capsule.checked_rows(m)
    for name, h in c["delivery_helper_sha256"].items():
        if capsule.sha((HERE / name).read_bytes()) != h:
            raise ValueError("delivery helper closure")
    t = release.Transport(
        os.environ["GH_TOKEN"],
        c["clock"],
        dict(
            id=404068972,
            tag="ufuk-cpu-ownplay-state-20261005",
            target_commitish="31a18e981e27e696fcca69c7e62210e7f0e89830",
            initial_assets=1,
        ),
    )
    inventory = t.inventory()
    for ref in c["protected_assets"]:
        matches = [r for r in inventory if r["name"] == ref["name"]]
        if len(matches) != 1 or matches[0]["size"] != ref["bytes"]:
            raise ValueError("old V6/V7/V8 assets must remain exact")
    raw_parts = []
    for row in c["parts"]:
        n = 4 * ((row["bytes"] + 2) // 3)
        url = release.API + "/repos/Eta06/HarbiChess/git/blobs/" + row["git_blob_sha1"]
        with t.request(url) as response:
            raw = git_bytes(response.read(n * 11 // 10 + 16384 + 1), row)
        proof = release.publish_verified(t, row["asset_name"], raw)
        if proof["sha256"] != row["sha256"] or not proof["anonymous_full_body_sha_verified"]:
            raise ValueError("public whole-part readback")
        raw_parts.append(raw)
    joined = b"".join(raw_parts)
    if len(joined) != c["full_capsule_bytes"] or capsule.sha(joined) != c["full_capsule_sha256"]:
        raise ValueError("exact unchanged complete capsule")
    originals = capsule.verify(joined, m)
    if len(originals) != c["original_files"] or len(originals) != 453:
        raise ValueError("all453 original files required")
    release.check_clock(c["clock"])
    print(
        json.dumps(
            dict(
                status="PASS-public-eight-parts-and-all453-original-byte-SHA",
                capsule_sha256=capsule.sha(joined),
                original_files=len(originals),
                strength_success=False,
                runtime_restore_claimed=False,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
