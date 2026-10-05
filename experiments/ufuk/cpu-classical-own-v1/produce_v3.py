"""One new classical own-experience segment, fixed snapshot, explicit V3 journal."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from journal_v3 import Actor, load_module, read, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "search-helper", "value-helper", "model", "output", "source-repo"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--target-actions", type=int, required=True)
    p.add_argument("--original-deadline", type=float, required=True)
    p.add_argument("--cpu-core", type=int, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--resume-sha256")
    a = p.parse_args()
    if a.cpu_core not in os.sched_getaffinity(0):
        raise ValueError("CPU affinity invalid")
    os.sched_setaffinity(0, {a.cpu_core})
    if sha(a.config) != a.config_sha256:
        raise ValueError("config SHA differs")
    c = json.loads(a.config.read_text())
    if a.original_deadline != c["original_deadline_epoch"]:
        raise ValueError("clock changed")
    if (
        sha(__file__) != c["runner_sha256"]
        or sha(Path(__file__).with_name("journal_v3.py")) != c["producer_sha256"]
    ):
        raise ValueError("runner/producer SHA differs")
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=a.source_repo, text=True
    ).strip() != c["source_commit"] or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=a.source_repo, text=True
    ):
        raise ValueError("clean source pin differs")
    if sha(Path(__file__).with_name("runtime.py")) != c["runtime_helper_sha256"]:
        raise ValueError("frozen runtime helper differs")
    if sha(a.model) != c["model_sha256"]:
        raise ValueError("frozen model differs")
    v = load_module(a.value_helper, c["value_helper_sha256"])
    evaluator = v.load_classical(a.model)
    prior = v.ClassicalValue()
    search = load_module(a.search_helper, c["search_helper_sha256"])

    def guard():
        from runtime import guard as resource_guard

        resource_guard(a.original_deadline, a.source_repo, a.output.parent)

    def factory():
        return search.BudgetSearch(
            evaluator.nonterminal, nodes=512, quiescence_plies=2, max_depth=8, guard=guard
        )

    state = None
    if a.resume is not None:
        if a.resume_sha256 is None or sha(a.resume) != a.resume_sha256:
            raise ValueError("resume SHA differs")
        state = read(a.resume)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    guard()
    actor = Actor(c, factory, prior.nonterminal, state)
    actor.advance(a.target_actions)
    guard()
    if sha(a.model) != c["model_sha256"]:
        raise ValueError("frozen model changed")
    journal_sha = save(a.output, actor.state)
    print(
        json.dumps(
            dict(
                schema="classical-own-qsearch-segment-receipt-v3",
                journal_sha256=journal_sha,
                actions=actor.state["actions"],
                resume_sha256=a.resume_sha256,
                original_deadline_epoch=a.original_deadline,
                old_actor_resume_claimed=False,
            )
        )
    )


if __name__ == "__main__":
    main()
