"""Read-only original4/5 terminal release proof; no profile/training entry point."""

import hashlib
from pathlib import Path

END = 1791180000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_table():
    table = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            pid = int(path.parent.name)
            table[pid] = dict(
                pid=pid,
                state=fields[0],
                ppid=int(fields[1]),
                pgid=int(fields[2]),
                startticks=int(fields[19]),
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return table


def live(identity, table):
    actual = table.get(identity["pid"])
    return bool(
        actual
        and actual["startticks"] == identity["startticks"]
        and (actual["state"] not in ("Z", "X"))
    )


def validate_barrier(config):
    from own8_failed4_dependency import remaining_owned
    from own8_failed4_dependency import validate_terminal as validate4
    from own8_failed5_terminal_barrier import bound, validate_release

    immutable = config["immutable_inputs"]
    for path, digest in immutable.items():
        assert sha(path) == digest

    def read(key):
        path = config[key]
        assert path in immutable
        return bound(path, immutable[path], sha)

    cohort = read("original_cohort")
    assert cohort["schema"] == "prospective-own45-completion-cohort-v1"
    assert cohort["completion_deadline_epoch"] == config["original_cohort_deadline_epoch"]
    terminal5 = read("failed5_terminal_receipt")
    release5 = read("failed5_owned_release")
    terminal5_sha = sha(config["failed5_terminal_receipt"])
    validate_release(release5, terminal5, terminal5_sha, sha)
    terminal4 = read("failed4_terminal_receipt")
    validate4(terminal4, cohort, sha)
    release4 = read("failed4_owned_release")
    assert release4["schema"] == "own4-incomplete-owned-compute-terminated-witness-v1"
    assert release4["terminal_receipt_sha256"] == sha(config["failed4_terminal_receipt"])
    assert release4["original_cohort_sha256"] == sha(config["original_cohort"])
    assert release4["all_registered_owned_groups_and_identities_terminated"] is True
    assert release4["method4_ready_latency_final_receipts_fabricated"] is False
    assert not remaining_owned(terminal4, process_table())
    assert not any(
        live(row, process_table()) for row in release4["terminated_tracked_pid_startticks"]
    )
    assert terminal5["original_training_started_epoch"] == terminal4["original_firstclock"]
    return {
        "failed5_terminal_sha256": terminal5_sha,
        "failed5_owned_release_sha256": sha(config["failed5_owned_release"]),
        "failed4_owned_release_sha256": sha(config["failed4_owned_release"]),
        "prior45_qualification_latency_final_receipts_substituted": False,
    }
