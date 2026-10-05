"""Map exact recorded remote hashes without reconstructing absent remote file bytes."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/workspace/HarbiChess")
WORK = Path("/workspace/work/harbichess")
STAGE = Path(__file__).resolve().parent
CONTROL = "/content/harbichess-fullgame-method2-inputs/method5-failed4-scheduling-v2"
PRODUCTION = "/content/harbichess-fullgame-method2-inputs/formal45-production"
known = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add(path, digest, purpose, reference, slots=(4, 5), local_exact=None):
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    entry = {
        "path": path,
        "expected_sha256": digest,
        "purpose": purpose,
        "slots": list(slots),
        "reviewed_public": True,
        "recorded_binding": {
            "local_receipt": str(reference),
            "local_receipt_sha256": sha(reference),
        },
    }
    if local_exact is not None and Path(local_exact).is_file() and sha(local_exact) == digest:
        entry["exact_local_copy"] = str(local_exact)
    if path in known:
        assert known[path]["expected_sha256"] == digest
    else:
        known[path] = entry


resume = WORK / "resume0051-remote-read.json"
resume_data = json.loads(json.loads(resume.read_text())["stdout"])
for path, item in resume_data["files"].items():
    if "startup428" in path:
        continue  # Different experiment; not needed for formal45 preservation.
    add(path, item["sha256"], "original-cohort-clock-owner-and-failed-baseline-control", resume)
probe = WORK / "method5-failed4-scheduling-v2/actual-terminal-binding-probe.json"
probe_data = json.loads(json.loads(probe.read_text())["stdout"])
for path, item in probe_data["files"].items():
    add(path, item["sha256"], "original-qualification-registration-baseline-or-prefix-audit", probe)
for slot, family in ((4, "ownsearch-method4"), (5, "search-acting-method5")):
    qpath = f"{PRODUCTION}/registration-{slot}/strength-bindings.json"
    q = probe_data["files"][qpath]["value"]
    for name, digest in q["helper_sha256"].items():
        local = ROOT / "experiments/ufuk" / family / name
        add(
            f"{PRODUCTION}/helpers{slot}/{name}",
            digest,
            "original-qualified-public-control-helper",
            probe,
            local_exact=local,
        )
    for name in ("registration.json", "strength-bindings.json"):
        remote = f"{PRODUCTION}/registration-{slot}/{name}"
        local = ROOT / "experiments/ufuk" / family / "frozen" / name
        if remote in known and local.is_file() and sha(local) == known[remote]["expected_sha256"]:
            known[remote]["exact_local_copy"] = str(local)
deployment = WORK / "method5-failed4-scheduling-v2/deployment-staged-result.json"
deployed = json.loads(json.loads(deployment.read_text())["stdout"])
for relative, digest in deployed["files"].items():
    if relative.startswith("analysis-v"):
        version, name = relative.split("/", 1)
        local = (
            ROOT
            / "experiments/ufuk"
            / version.replace("analysis-v", "analysis-repair-v")
            / "method5"
            / name
        )
    else:
        local = ROOT / "experiments/ufuk/method5-failed4-scheduling-v2" / relative
    add(
        f"{CONTROL}/{relative}",
        digest,
        "failed4-terminal-supplement-or-versioned-analysis-control",
        deployment,
        local_exact=local,
    )
activation = WORK / "method5-failed4-scheduling-v2/activation-actual-result.json"
activated = json.loads(json.loads(activation.read_text())["stdout"])
add(
    f"{CONTROL}/owned4-stop-result.json",
    activated["owner"]["stop4_result_sha256"],
    "actual-owned4-terminal-stop-pid-and-durable-native-inventory",
    activation,
)
# Newly recorded actual file SHA bindings; embedded values are not reserialized.
for basename in (
    "method5-native-audit-owner-state-0130.json",
    "method5-postfail-ownership-actual-0126.json",
):
    reference = WORK / basename
    data = json.loads(json.loads(reference.read_text())["stdout"])
    for path, item in data["files"].items():
        if "own-terminal-source428" in path:
            continue  # Separate development waiter, not a formal45 archive input.
        if "sha256" in item:
            add(path, item["sha256"], "actual-final-native-audit-or-owned-control", reference)
        else:
            for name, record in item.items():
                if isinstance(record, dict) and "sha256" in record:
                    add(
                        f"{path}/{name}",
                        record["sha256"],
                        "actual-failed5-replay-metadata-no-promotion",
                        reference,
                    )
local_barrier = WORK / "method5-terminal-release-v3"
remote_barrier = "/content/harbichess-fullgame-method2-inputs/method5-terminal-release-v3"
for name in (
    "terminal.json",
    "owned-release.json",
    "owned-inventory-before.json",
    "owned-inventory-after.json",
):
    local = local_barrier / name
    add(
        f"{remote_barrier}/{name}",
        sha(local),
        "actual-failed5-terminal-owned-release",
        local,
        local_exact=local,
    )
terminal = json.loads((local_barrier / "terminal.json").read_bytes())
release = json.loads((local_barrier / "owned-release.json").read_bytes())
for item in terminal["owner_identities"]:
    if "terminated_owner_receipt" in item:
        add(
            item["terminated_owner_receipt"],
            item["terminated_owner_receipt_sha256"],
            "actual-terminated5-train-audit-baseline-owner",
            local_barrier / "terminal.json",
        )
# These filenames are established by executed control code; their original byte
# digests are absent locally. Never invent a digest by reserializing embedded JSON.
unknown = [
    {
        "path": f"{CONTROL}/{name}",
        "expected_sha256": None,
        "purpose": purpose,
        "slots": [4, 5],
        "reviewed_public": False,
        "requires_root_remote_byte_review": True,
    }
    for name, purpose in (
        ("owned4-stop-inventory-before.json", "prestop-owned4-inventory"),
        ("control-plan-result.json", "actual-plan-only-validation"),
        ("control-owner.json", "actual-supplemental-owner-identity"),
        ("activation.json", "actual-original-helper-and-failed4-activation-binding"),
    )
]
post = "/content/harbichess-runs/search-acting-method5-posttraining-scheduling-v2"
for relative in (
    "failure.json",
    "replay-20261525-command.json",
):
    unknown.append(
        {
            "path": f"{post}/{relative}",
            "expected_sha256": None,
            "purpose": "actual-failed5-replay-and-preserved-posttraining-control",
            "slots": [4, 5],
            "reviewed_public": False,
            "requires_root_remote_byte_review": True,
        }
    )
request = {
    "schema": "formal45-evidence-resolution-request-v1",
    "status": "known-recorded-bindings-plus-explicit-unresolved-no-fabricated-hashes",
    "known": sorted(known.values(), key=lambda row: row["path"]),
    "unresolved": [row for row in unknown if row["path"] not in known],
    "root_additional_required_roles": [
        "Actual failed5 terminal receipt and owned-release receipt (paths+SHAs supplied by ROOT)",
        "All real failed5 replay partial native/model/Adam/RNG/journal artifacts, exact files "
        "identified by ROOT read-only inventory; mark failed-partial, never closed-native",
        "Any remaining original failed baseline426 partial markers, exact paths and SHA",
        "Original source/control Git bundles beyond automatically-created producer bundles",
    ],
    "no_outcome_or_model_queries": True,
    "no_remote_execution": True,
    "observed_absent_not_fabricated": [post + "/replay-20261525-process-result.json"],
}
(STAGE / "evidence-resolution-request-v2.json").write_text(json.dumps(request, indent=2) + "\n")
spec = json.loads((STAGE / "backup-spec-DRAFT.json").read_text())
spec["status"] = "DRAFT-pending-failed5-terminal-release-and-remote-evidence-review"
spec.pop("latency_barrier", None)
spec["backup_barrier"] = {
    "kind": "failed5-terminal-owned-release-v1",
    "terminal_schema": terminal["schema"],
    "terminal_status": terminal["status"],
    "release_schema": release["schema"],
    "before_inventory_sha256": release["before_inventory_sha256"],
    "after_inventory_sha256": release["after_inventory_sha256"],
    "terminal_path": remote_barrier + "/terminal.json",
    "terminal_sha256": sha(local_barrier / "terminal.json"),
    "release_path": remote_barrier + "/owned-release.json",
    "release_sha256": sha(local_barrier / "owned-release.json"),
    "source_commit": "4515a7c0dda3b4f9615c2fc78a47c872ab14699d",
    "original_registration_sha256": (
        "a9ae975e6ecb486f964ee0514d98fb4bc9d50d3b949ecabd0cf1e441bd75d271"
    ),
    "original_qualification_config_sha256": (
        "5fa34fdc69a49b78fecae17ee9ade86da994c548bbda81473d86c454c6cc733a"
    ),
}
for job in spec["jobs"].values():
    job["reviewed_public_evidence"] = [
        {
            "path": item["path"],
            "sha256": item["expected_sha256"],
            "purpose": item["purpose"],
            "reviewed_public": True,
        }
        for item in request["known"]
    ]
spec["unresolved_evidence_request_sha256"] = sha(STAGE / "evidence-resolution-request-v2.json")
(STAGE / "backup-spec-known-evidence-v2-PENDING.json").write_text(json.dumps(spec, indent=2) + "\n")
print(
    json.dumps(
        {
            "known_file_count": len(known),
            "known_original_hashes": True,
            "unresolved_file_count": len(request["unresolved"]),
            "request_sha256": sha(STAGE / "evidence-resolution-request-v2.json"),
            "pending_spec_sha256": sha(STAGE / "backup-spec-known-evidence-v2-PENDING.json"),
        }
    )
)
