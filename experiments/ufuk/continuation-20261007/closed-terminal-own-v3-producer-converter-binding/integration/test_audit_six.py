import hashlib
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import chess
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_six  # noqa: E402


def write(path, raw):
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def fixture(tmp_path):
    board = chess.Board()
    alias = int.from_bytes(
        hashlib.sha256(min(board.board_fen(), board.mirror().board_fen()).encode()).digest()[:8],
        "little",
        signed=True,
    )
    selected = [f"r{i}:0" for i in audit_six.ORDINALS]
    ids = [f"r{i}:0" for i in range(1024)]
    side = tmp_path / "search-aliases-000000.bin"
    side_ref = write(side, struct.pack("<1024q", *([alias] * 1024)))
    events_path = tmp_path / "events.jsonl"
    events = []
    for i in range(1024):
        row = {
            "root_id": f"r{i}",
            "root_ordinal": i,
            "root_fen": chess.STARTING_FEN,
            "root_prefix_uci": [],
            "local_ply": 0,
            "history_uci": [],
            "fen4": " ".join(board.fen().split()[:4]),
            "mover": "white",
            "root_alias": alias,
            "raw_q_mover": 0.25,
            "clipped_q_mover": 0.25,
            "mate_range_score_returned": False,
            "selected_best_uci": "g1f3",
            "nodes": 22,
            "evaluations": 1,
            "completed_depth": 1,
            "root_actions": 20,
            "actual_eval_calls": 1,
            "search_alias_ref": {"file": side.name, "offset_bytes": 8 * i, "count": 1},
        }
        events.append({"type": "search_row", "row": row})
    raw = b"".join(
        json.dumps(e, sort_keys=True, separators=(",", ":")).encode() + b"\n" for e in events
    )
    event_ref = write(events_path, raw)
    receipt = {
        "schema": "own-nnue-closed-terminal-collection-receipt-v1",
        "seed": 20262905,
        "training_row_ids": ids,
        "periodic_independent_search_rows": selected,
        "alias_chunks": [
            {"file": side.name, "bytes": side.stat().st_size, "sha256": side_ref["sha256"]}
        ],
    }
    spec = {
        "registration": {"path": str(tmp_path / "registration.json"), "sha256": "a" * 64},
        "receipt": {"path": str(tmp_path / "receipt.json"), "sha256": "b" * 64},
        "events": event_ref,
        "alias_chunks": {side.name: side_ref},
    }
    # Receipt SHA is compared by fake converter only; write stable local inputs.
    (tmp_path / "registration.json").write_text("{}")
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    spec["registration"]["sha256"] = hashlib.sha256(
        (tmp_path / "registration.json").read_bytes()
    ).hexdigest()
    spec["receipt"]["sha256"] = hashlib.sha256((tmp_path / "receipt.json").read_bytes()).hexdigest()
    return spec, receipt, alias


class FakeCollector:
    @staticmethod
    def alias(board):
        return int.from_bytes(
            hashlib.sha256(min(board.board_fen(), board.mirror().board_fen()).encode()).digest()[
                :8
            ],
            "little",
            signed=True,
        )

    class Traced:
        def __init__(self, fn):
            self.fn, self.seen, self.calls = fn, set(), 0

        def __call__(self, board):
            self.seen.add(FakeCollector.alias(board))
            self.calls += 1
            return self.fn(board)


class FakeSearch:
    class BudgetSearch:
        def __init__(self, evaluator, **kwargs):
            self.evaluator = evaluator

        def search(self, board):
            self.evaluator(board)
            return SimpleNamespace(
                move=chess.Move.from_uci("g1f3"),
                value=0.25,
                nodes=22,
                evaluations=1,
                completed_depth=1,
                root_actions=20,
            )


class FakeConverter:
    @staticmethod
    def convert(spec, guard):
        provenance = {
            "schema": "NNUE-own1024-closed-terminal-data-provenance-v1",
            "collection_receipt_sha256": spec["receipt"]["sha256"],
        }
        return b"synthetic-data", json.dumps(provenance).encode()


def test_six_packets_replay_and_bind_aliases(tmp_path):
    spec, receipt, _ = fixture(tmp_path)
    result = audit_six.audit_six(
        spec,
        receipt,
        converter=FakeConverter,
        collector=FakeCollector,
        producer=None,
        search=FakeSearch,
        evaluator=lambda _board: 0.0,
        clock={"sha256": "c" * 64, "first": 1, "deadline": 100},
    )
    assert result["schema"] == audit_six.AUDIT_SCHEMA
    assert result["status"] == audit_six.AUDIT_STATUS
    assert [x["row_id"] for x in result["packets"]] == [
        receipt["training_row_ids"][i] for i in audit_six.ORDINALS
    ]
    assert result["new_games"] == result["optimizer_updates"] == 0


def test_six_packet_rejects_resealed_search_mismatch(tmp_path):
    spec, receipt, _ = fixture(tmp_path)
    events = Path(spec["events"]["path"])
    all_rows = [json.loads(line) for line in events.read_text().splitlines()]
    all_rows[0]["row"]["raw_q_mover"] = 0.5
    raw = b"".join(
        json.dumps(e, sort_keys=True, separators=(",", ":")).encode() + b"\n" for e in all_rows
    )
    spec["events"] = write(events, raw)
    with pytest.raises(ValueError, match="packet differs"):
        audit_six.audit_six(
            spec,
            receipt,
            converter=FakeConverter,
            collector=FakeCollector,
            producer=None,
            search=FakeSearch,
            evaluator=lambda _board: 0.0,
            clock={"sha256": "c" * 64, "first": 1, "deadline": 100},
        )


def test_six_packet_rejects_alias_trace_mismatch(tmp_path):
    spec, receipt, _ = fixture(tmp_path)
    path = Path(next(iter(spec["alias_chunks"].values()))["path"])
    raw = bytearray(path.read_bytes())
    raw[:8] = struct.pack("<q", 123456)
    ref = write(path, bytes(raw))
    spec["alias_chunks"][path.name] = ref
    receipt["alias_chunks"][0]["sha256"] = ref["sha256"]
    Path(spec["receipt"]["path"]).write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="alias replay"):
        audit_six.audit_six(
            spec,
            receipt,
            converter=FakeConverter,
            collector=FakeCollector,
            producer=None,
            search=FakeSearch,
            evaluator=lambda _board: 0.0,
            clock={"sha256": "c" * 64, "first": 1, "deadline": 100},
        )
