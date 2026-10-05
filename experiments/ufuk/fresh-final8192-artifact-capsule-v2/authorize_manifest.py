"""ROOT-reviewed approval binds a single exact local manifest; no arbitrary paths."""

import json
from pathlib import Path

import artifact_capsule as c


def authorize(raw, approval):
    c.require(
        approval.get("schema") == "fresh8192-root-manifest-approval-v2"
        and approval.get("status") == "ROOT-reviewed-prospectively-approved",
        "approval",
    )
    digest = c.sha(raw)
    c.require(digest == approval["manifest_sha256"], "approved-manifest-sha")
    m = json.loads(raw)
    c.validate(m)
    c.require(c.source_seals(m) == approval["source_seals"], "approved-source-seals")
    c.require(m["producer_bindings"] == approval["producer_bindings"], "approved-actor-anchors")
    e8 = "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
    c.require(
        [b["seed"] for b in m["producer_bindings"]] == [20262805, 20262806]
        and all(
            b["actions"] == 8192
            and b["model_sha256"] == e8
            and b["journal_schema"] == "fresh-qsearch-selfplay-journal-v2"
            for b in m["producer_bindings"]
        ),
        "four-file-milestone-anchors",
    )
    return {digest: "manifest-8192-DRAFT.json"}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    approval = json.loads((root / "ROOT-approval-PENDING.json").read_bytes())
    print(
        json.dumps(
            authorize((root / "manifest-8192-DRAFT.json").read_bytes(), approval), sort_keys=True
        )
    )
