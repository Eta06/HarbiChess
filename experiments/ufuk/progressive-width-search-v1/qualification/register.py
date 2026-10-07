"""Metadata-only ROOT-stamped registration; never launches a child/search."""

import argparse
import json
from pathlib import Path

import develop


def reference(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=develop.sha(path))


def build(args):
    reg = dict(
        schema="progressive-width-human-prior-development-registration-v1",
        status="registered",
        mode=args.mode,
        first=args.first,
        deadline=args.deadline,
        operator_end_epoch=args.operator_end_epoch,
        cpu_core=args.cpu_core,
        root_width="progressive",
        advanced_incheck_extensions=args.incheck_extensions,
        search_math=dict(nodes=512, qdepth=2, max_depth=8),
        core_repo=str(args.core_repo.resolve()),
        core_commit=args.core_commit,
        output=str(args.output.resolve()),
        helper_sha256=develop.sha(develop.__file__),
        old_search=reference(args.old_search),
        advanced_search=reference(args.advanced_search),
        prior_helper=reference(args.prior_helper),
    )
    for name in ("train_roots", "book", "stockfish", "profile_result"):
        path = getattr(args, name)
        if path is not None:
            reg[name] = reference(path)
    develop.clock(reg, args.first)
    if args.mode == "profile":
        develop.prepare_profile_roots(reg)
    elif any(name not in reg for name in ("book", "stockfish", "profile_result")):
        raise ValueError("arena requires known8 book + actual profile + official SF")
    if not args.output.resolve().is_relative_to("/dev/shm"):
        raise ValueError("RAM output only")
    return reg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("profile", "arena"), required=True)
    for name in ("first", "deadline", "operator-end-epoch"):
        p.add_argument("--" + name, type=float, required=True)
    p.add_argument("--cpu-core", type=int, required=True)
    p.add_argument("--core-repo", type=Path, required=True)
    p.add_argument("--core-commit", required=True)
    for name in ("old-search", "advanced-search", "prior-helper", "output", "registration"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("train-roots", "book", "stockfish", "profile-result"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--incheck-extensions", type=int, choices=(0, 1), default=1)
    a = p.parse_args()
    a.registration.parent.mkdir(parents=True, exist_ok=True)
    with a.registration.open("xb") as f:
        f.write(develop.canonical(build(a)) + b"\n")
    print(json.dumps(dict(registration=str(a.registration), sha256=develop.sha(a.registration))))


if __name__ == "__main__":
    main()
