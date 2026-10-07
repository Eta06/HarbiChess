"""Rule-only procedural TRAIN source; never imports a model or evaluator."""

import argparse
import hashlib
import json
import random
import struct
import time
from pathlib import Path

import chess

SCHEMA = "human-procedural-legal-walk-selection-v2"
POOL = "human-randomstarts-train-roots-v2"
RECEIPT = "human-procedural-root-bank-receipt-v2"
RECIPE = dict(
    count=4096,
    max_attempts=16384,
    min_plies=6,
    max_plies=24,
    legal_order="sorted-UCI",
    length="uniform-inclusive-before-walk",
    move="uniform-legal-at-each-ply",
    duplicate="final-conservative-alias",
    protection="final-conservative-alias-only",
    terminal="claim_draw=True",
)


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    p = Path(path).resolve(strict=True)
    return dict(path=str(p), sha256=sha(p))


def pinned(r):
    p = Path(r["path"])
    if not p.is_file() or p.is_symlink() or sha(p) != r["sha256"]:
        raise ValueError("immutable regular source binding")
    return p


def alias(board):
    b = min(board.board_fen(), board.mirror().board_fen()).encode("ascii")
    return int.from_bytes(hashlib.sha256(b).digest()[:8], "little", signed=True)


def protection(r):
    data = pinned(r).read_bytes()
    if len(data) > 256 * 1024**2 or len(data) % 8:
        raise ValueError("bounded canonical int64 protection")
    values = [x[0] for x in struct.iter_unpack("<q", data)]
    if values != sorted(set(values)):
        raise ValueError("sorted unique protection")
    return set(values)


class BankExhausted(ValueError):
    def __init__(self, selection, trace):
        super().__init__("attempt cap exhausted; no relaxation or alternate seed")
        self.selection, self.trace = selection, trace


def generate(seed, protected, recipe=None, guard=lambda: None):
    """Configurable counts only for synthetic fixtures; production recipe is fixed."""
    recipe = dict(RECIPE if recipe is None else recipe)
    if (
        type(seed) is not int
        or recipe["count"] <= 0
        or recipe["max_attempts"] < recipe["count"]
        or not 1 <= recipe["min_plies"] <= recipe["max_plies"] <= 24
    ):
        raise ValueError("bounded walk recipe")
    rng = random.Random(seed)
    before = hashlib.sha256(canonical(rng.getstate())).hexdigest()
    rows, trace, seen = [], [], set()
    for attempt in range(recipe["max_attempts"]):
        guard()
        desired = rng.randint(recipe["min_plies"], recipe["max_plies"])
        board, moves = chess.Board(), []
        walk_aliases = [alias(board)]
        reason = None
        for _ in range(desired):
            if board.outcome(claim_draw=True) is not None:
                reason = "terminal-before-length"
                break
            legal = sorted(board.legal_moves, key=lambda m: m.uci())
            if not legal:
                reason = "no-legal-move"
                break
            move = legal[rng.randrange(len(legal))]
            moves.append(move.uci())
            board.push(move)
            walk_aliases.append(alias(board))
        a = alias(board)
        if reason is None:
            if not board.is_valid() or board.outcome(claim_draw=True) is not None:
                reason = "terminal-or-invalid-final"
            elif a in protected:
                reason = "protected-final-alias"
            elif a in seen:
                reason = "duplicate-final-alias"
        identifier = f"procwalk-v2:{seed}:{attempt:05d}"
        history_sha = hashlib.sha256(
            (chess.STARTING_FEN + "\n" + " ".join(moves)).encode()
        ).hexdigest()
        event = dict(
            attempt=attempt,
            row_id=identifier,
            desired_plies=desired,
            walk_aliases=walk_aliases,
            interior_protected_aliases=sorted(set(walk_aliases[:-1]) & protected),
            prefix_uci=moves,
            final_fen=board.fen(),
            final_alias=a,
            history_sha256=history_sha,
            status=reason or "accepted",
        )
        trace.append(event)
        if reason is None:
            seen.add(a)
            rows.append(
                dict(
                    row_id=identifier,
                    root_id=identifier,
                    source_row_id=identifier,
                    trajectory_id=identifier,
                    root_fen=chess.STARTING_FEN,
                    prefix_uci=moves,
                    role="TRAIN",
                    procedural_attempt=attempt,
                    final_fen=board.fen(),
                    final_alias=a,
                    history_sha256=history_sha,
                    walk_aliases=walk_aliases,
                )
            )
        if len(rows) == recipe["count"]:
            break
    selection = dict(
        schema=SCHEMA,
        seed=seed,
        recipe=recipe,
        synthetic_only=recipe != RECIPE,
        rows=rows,
        rng_before_sha256=before,
        rng_after_sha256=hashlib.sha256(canonical(rng.getstate())).hexdigest(),
    )
    if len(rows) != recipe["count"]:
        raise BankExhausted(selection, trace)
    return selection, trace


def verify(bundle, expected_protected=None, guard=lambda: None):
    """Replay the generator and every rejection; hash linkage alone is insufficient."""
    receipt = json.loads(pinned(bundle).read_bytes())
    if (
        receipt["schema"] != RECEIPT
        or receipt["status"] != "PASS-rule-only-procedural-root-bank"
        or receipt["synthetic_only"] is not False
        or receipt["recipe"] != RECIPE
        or receipt["generator_sha256"] != sha(__file__)
        or receipt["chess_version"] != chess.__version__
    ):
        raise ValueError("production recipe/generator/runtime binding")
    if expected_protected is not None and receipt["protected_aliases"] != expected_protected:
        raise ValueError("same protected source")
    if (
        receipt["teacher_labels_used"] is not False
        or not receipt["first"] < receipt["finished_epoch"] <= receipt["deadline"]
        or receipt["deadline"]
        > min(receipt["first"] + 600, receipt["operator_end_epoch"], 1791448916.685839)
    ):
        raise ValueError("complete prospective procedural source clock")
    registration = json.loads(pinned(receipt["registration"]).read_bytes())
    if (
        registration["schema"] != "human-procedural-root-bank-registration-v2"
        or registration["status"] != "registered"
        or any(
            registration[k] != receipt[k]
            for k in (
                "seed",
                "recipe",
                "generator_sha256",
                "protected_aliases",
                "first",
                "deadline",
                "operator_end_epoch",
            )
        )
    ):
        raise ValueError("exact ROOT source registration")
    selected, trace = generate(
        receipt["seed"], protection(receipt["protected_aliases"]), guard=guard
    )
    if pinned(receipt["selection"]).read_bytes() != canonical(selected):
        raise ValueError("selected history/order/source tampering")
    if pinned(receipt["attempt_trace"]).read_bytes() != b"".join(canonical(x) for x in trace):
        raise ValueError("discard trace tampering")
    if receipt["accepted_count"] != 4096 or receipt["attempt_count"] != len(trace):
        raise ValueError("complete procedural bank counts")
    return selected, receipt


def build(spec, output):
    if (
        spec["schema"] != "human-procedural-root-bank-registration-v2"
        or spec["status"] != "registered"
    ):
        raise ValueError("ROOT prospective bank registration")
    first, end, op = spec["first"], spec["deadline"], spec["operator_end_epoch"]
    if not first <= time.time() < end <= min(first + 600, op, 1791448916.685839):
        raise ValueError("new bounded source clock")
    if spec["generator_sha256"] != sha(__file__) or spec["recipe"] != RECIPE:
        raise ValueError("exact source and declared recipe")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "registration.json").open("xb") as f:
        f.write(canonical(spec))

    def guard():
        if time.time() >= end:
            raise TimeoutError("original bank deadline")

    try:
        selection, trace = generate(
            spec["seed"], protection(spec["protected_aliases"]), guard=guard
        )
    except BankExhausted as error:
        with (directory / "failed-attempts.jsonl").open("xb") as f:
            f.write(b"".join(canonical(x) for x in error.trace))
        with (directory / "failure.json").open("xb") as f:
            f.write(
                canonical(
                    dict(
                        schema=RECEIPT,
                        status="FAIL-attempt-cap",
                        seed=spec["seed"],
                        accepted_count=len(error.selection["rows"]),
                        attempt_count=len(error.trace),
                        recipe=RECIPE,
                        first=first,
                        deadline=end,
                        operator_end_epoch=op,
                    )
                )
            )
        raise
    for name, data in [
        ("selection.json", canonical(selection)),
        ("attempts.jsonl", b"".join(canonical(x) for x in trace)),
    ]:
        with (directory / name).open("xb") as f:
            f.write(data)
    result = dict(
        schema=RECEIPT,
        status="PASS-rule-only-procedural-root-bank",
        seed=spec["seed"],
        recipe=RECIPE,
        synthetic_only=False,
        generator_sha256=sha(__file__),
        chess_version=chess.__version__,
        protected_aliases=spec["protected_aliases"],
        registration=ref(directory / "registration.json"),
        selection=ref(directory / "selection.json"),
        attempt_trace=ref(directory / "attempts.jsonl"),
        accepted_count=4096,
        attempt_count=len(trace),
        first=first,
        deadline=end,
        operator_end_epoch=op,
        finished_epoch=time.time(),
        teacher_labels_used=False,
        scope="conditional uniform legal walks, not uniform chess positions; common START ancestry",
    )
    guard()
    with (directory / "receipt.json").open("xb") as f:
        f.write(canonical(result))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    build(json.loads(a.registration.read_bytes()), a.output)
