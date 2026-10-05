"""Tiny subprocess fixture driver; not real Qsearch, experience, or qualification."""

import argparse
import json
from pathlib import Path

from journal_v3 import Actor, read, save
from learner import Learner
from test_pipeline import FixtureSearch, config
from value import ClassicalValue


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["actor", "learner"], required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.mode == "actor":
        actor = Actor(
            config(),
            FixtureSearch,
            ClassicalValue().nonterminal,
            read(a.resume) if a.resume else None,
        )
        actor.advance(a.stop)
        save(a.output, actor.state)
    else:
        contract = {"updates": 8, "schema": "synthetic-offline-not-data-qualified"}
        learner = Learner(3, contract, json.loads(a.resume.read_text()) if a.resume else None)
        groups = {"a": [((1.0,) * 18, 0.1, 1.0, 0.5)], "b": [((-0.5,) * 18, -0.2, -1.0, -0.7)]}
        learner.advance(groups, a.stop)
        a.output.write_text(json.dumps(learner.native(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
