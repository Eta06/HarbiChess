"""Append one actual closed source7 archive to byte-preserved prior127 rows."""

import argparse
import hashlib
import json
import re
from pathlib import Path

PRIOR_SHA = "a3874ea7f91096783e5445f0d6486beb0b75ec3a5f1f360d5f78708403a24841"


def extend(prior_bytes, result, base_url):
    if hashlib.sha256(prior_bytes).hexdigest() != PRIOR_SHA:
        raise ValueError("original127-manifest-not-exact")
    obj = json.loads(prior_bytes)
    if result["status"] != "closed-source7-explicit-archive-pack-pass":
        raise ValueError("closed-source7-pack-PASS-required")
    if result["source_commit"] != "c022bc1605b44c3089439da5c6efb7bd4db4ff81":
        raise ValueError("source7-original-producer-required")
    if not re.fullmatch(r"https://[a-z0-9-]+\.trycloudflare\.com", base_url):
        raise ValueError("fixed-authorized-quicktunnel-host-required")
    digest = result["sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or not 0 < result["bytes"] <= 1024**3:
        raise ValueError("archive-sha-or-size")
    name = f"sha256-{digest}.tar.gz"
    if any(row["name"] == name for row in obj["assets"]):
        raise ValueError("duplicate-new-asset")
    previous = list(obj["assets"])
    obj["assets"].append(
        {"name": name, "bytes": result["bytes"], "sha256": digest, "url": base_url + "/" + name}
    )
    if len(obj["assets"]) > 200 or sum(r["bytes"] for r in obj["assets"]) > 8 * 1024**3:
        raise ValueError("unchanged-aggregate-caps-exceeded")
    assert obj["assets"][:-1] == previous
    return obj


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("prior", "pack-result", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--base-url", required=True)
    a = p.parse_args()
    with a.output.open("x") as f:
        json.dump(
            extend(a.prior.read_bytes(), json.loads(a.pack_result.read_text()), a.base_url),
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
