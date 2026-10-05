import hashlib
import json
import shlex
from pathlib import Path

import formal45_remote_backup_owner_v2 as owner
import freeze_reviewed_spec as freeze
import pytest
import remote_evidence_review as review


def fixtures():
    root = Path(__file__).resolve().parent / "fixtures"
    terminal = json.loads((root / "terminal.json").read_bytes())
    release = json.loads((root / "owned-release.json").read_bytes())
    barrier = {
        "kind": "failed5-terminal-owned-release-v1",
        "source_commit": owner.SOURCE5,
        "original_registration_sha256": owner.REGISTRATION5,
        "terminal_sha256": hashlib.sha256((root / "terminal.json").read_bytes()).hexdigest(),
        "release_sha256": hashlib.sha256((root / "owned-release.json").read_bytes()).hexdigest(),
        "terminal_schema": terminal["schema"],
        "terminal_status": terminal["status"],
        "release_schema": release["schema"],
        "before_inventory_sha256": release["before_inventory_sha256"],
        "after_inventory_sha256": release["after_inventory_sha256"],
    }
    return barrier, terminal, release


def test_failed5_receipt_never_claims_genuine_ready_or_latency():
    result = owner.validate_failed5_barrier(*fixtures())
    assert result["genuine_latency_claim"] is False
    assert "terminal-owned-release-no-promotion" in result["status"]


@pytest.mark.parametrize(
    "mutant",
    [
        "source",
        "registration",
        "qualification",
        "clock",
        "eligible",
        "owned-live",
        "wrong-terminal-link",
    ],
)
def test_failure_release_requires_exact_original_bindings(mutant):
    barrier, terminal, release = fixtures()
    if mutant == "source":
        terminal["source_commit"] = "c" * 40
    elif mutant == "registration":
        barrier["original_registration_sha256"] = "c" * 64
    elif mutant == "qualification":
        terminal["original_qualification_config_sha256"] = "c" * 64
    elif mutant == "clock":
        terminal["original_training_started_epoch"] += 1
    elif mutant == "eligible":
        terminal["no_strength_qualification_or_model_promotion"] = False
    elif mutant == "owned-live":
        release["all_registered_owned_groups_and_identities_terminated"] = False
    else:
        release["terminal_receipt_sha256"] = "c" * 64
    with pytest.raises(AssertionError):
        owner.validate_failed5_barrier(barrier, terminal, release)


def test_read_pinned_receipt_checks_original_bytes_and_rejects_link(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_bytes(b'{"status":"complete"}\n')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert owner._read_pinned_receipt(path, sha) == {"status": "complete"}
    path.write_bytes(b'{"status":"complete"} ')
    with pytest.raises(AssertionError):
        owner._read_pinned_receipt(path, sha)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(AssertionError):
        owner._read_pinned_receipt(link, hashlib.sha256(path.read_bytes()).hexdigest())


def test_resolution_shell_argument_never_executes_request_data():
    request = {"known": [{"path": "x$(touch bad);`secret`"}], "unresolved": []}
    argv = shlex.split(review.command_for(request))
    assert argv[:2] == ["python3", "-c"] and argv[2] == review.REMOTE_PROGRAM and len(argv) == 4
    import base64

    assert json.loads(base64.b64decode(argv[3])) == request


def test_approval_cannot_freeze_missing_or_unapproved_original_digest():
    pending = {"jobs": {"20261425": {}}}
    request = {
        "schema": "formal45-evidence-resolution-request-v1",
        "known": [{"path": "original", "expected_sha256": "a" * 64, "purpose": "public-original"}],
        "unresolved": [],
    }
    response = {
        "schema": "formal45-exact-file-review-v1",
        "records": [{"path": "original", "status": "exact-regular-file-read", "sha256": "b" * 64}],
    }
    approval = {
        "schema": "formal45-public-evidence-byte-approval-v1",
        "status": "approved-exact-public-byte-allowlist",
        "reviewed_public_sha256": {"original": "b" * 64},
    }
    with pytest.raises(AssertionError):
        freeze.freeze(pending, request, response, approval)


@pytest.mark.parametrize("phase", ["before", "after"])
def test_actual_owned_inventory_is_empty_and_rejects_a_live_owner(phase):
    barrier, _, _ = fixtures()
    root = Path(__file__).resolve().parent / "fixtures"
    inventory = json.loads((root / f"owned-inventory-{phase}.json").read_bytes())
    owner.validate_owned_inventory(barrier, inventory)
    inventory["remaining_owned_pid_startticks"] = [{"pid": 104294, "startticks": 2286865}]
    with pytest.raises(AssertionError):
        owner.validate_owned_inventory(barrier, inventory)


def test_existing_serialized_transport_receives_fixed_bounded_arguments():
    class Mock:
        def call(self, command, **kwargs):
            assert command == "exact-safe-command"
            assert kwargs == {"timeout": 120, "capture_output": True}
            return "synthetic-transport-response"

    assert review.bounded_call(Mock(), "exact-safe-command") == "synthetic-transport-response"


def test_transport_error_suppresses_connection_details():
    class Mock:
        def call(self, command, **kwargs):
            raise TimeoutError("synthetic-secret-connection-detail")

    with pytest.raises(RuntimeError) as caught:
        review.bounded_call(Mock(), "exact-safe-command")
    assert "synthetic-secret" not in str(caught.value)
    assert caught.value.__suppress_context__ is True
