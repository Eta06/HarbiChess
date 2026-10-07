"""Synthetic rules/ledger integration only: no real planner, NN forward or SGD."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import chess
import pytest
from contracts import clock
from convert import canonical, convert, sha

PRODUCER = Path("/workspace/work/harbichess/continuation-20261007/nnue-own-producer-v2")


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def write(path, value):
    path.write_bytes(canonical(value))
    return ref(path)


@pytest.fixture(scope="module")
def ledger(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic-own-ledger")
    # Use independently frozen test copy of production's metadata/rules collector.
    producer = d / "producer"
    producer.mkdir()
    for name in ("collector.py", "run_collection.py"):
        (producer / name).write_bytes((PRODUCER / name).read_bytes())
    spec = importlib.util.spec_from_file_location(
        "synthetic_own_collector", producer / "collector.py"
    )
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    rows, queue = [], [[]]
    while len(rows) < 4096:
        prefix = queue.pop(0)
        board = chess.Board()
        for uci in prefix:
            board.push_uci(uci)
        if board.outcome(claim_draw=True) is None:
            rows.append(
                dict(
                    root_id=f"synthetic-{len(rows)}",
                    root_fen=chess.STARTING_FEN,
                    prefix_uci=prefix,
                    role="TRAIN",
                    trajectory_id="synthetic-game",
                    source_row_id=f"synthetic-{len(rows)}",
                )
            )
            queue.extend([*prefix, m.uci()] for m in board.legal_moves)
    selection_ref = write(
        d / "selection.json",
        dict(
            rows=[
                dict(
                    row_id=r["root_id"],
                    trajectory_id=r["trajectory_id"],
                    root_fen=r["root_fen"],
                    prefix_uci=r["prefix_uci"],
                )
                for r in rows
            ]
        ),
    )
    labels_path = d / "synthetic-labels.json.gz"
    labels_path.write_bytes(b"synthetic-source-marker-no-teacher-search")
    labels_ref = ref(labels_path)
    pool = dict(
        schema=c.POOL,
        selection_status="pass",
        train_only=True,
        rows=rows,
        source_selection_sha256=selection_ref["sha256"],
        source_teacher_labels_sha256=labels_ref["sha256"],
    )
    writer = c.AliasWriter(d)
    events = []

    def emit(event):
        if event["type"] == "search_row":
            event["row"]["search_alias_ref"] = writer.append(event["row"].pop("eval_aliases"))
        events.append(event)

    class FakeSearch:
        def __init__(self, fn):
            self.fn = fn

        def search(self, board):
            self.fn(board)  # Only mock scalar; zero network/search work.
            moves = sorted(board.legal_moves, key=lambda m: m.uci())
            return SimpleNamespace(
                move=moves[0],
                value=0.2,
                nodes=len(moves) + 1,
                root_actions=len(moves),
                evaluations=1,
                completed_depth=0,
            )

    result = c.collect(
        pool,
        seed=20262905,
        protected=set(),
        factory=SimpleNamespace(evaluator=lambda _: 0.0, make=FakeSearch),
        on_event=emit,
    )
    writer.close()
    feature = d / "model.py"
    feature.write_text("def board_indices(b): return [0]\n")
    prior = d / "prior.py"
    prior.write_text("PRIOR=(0.0,)\nSCALE=600\ndef features(b): return (1.0,)\n")
    protected = d / "protected.bin"
    protected.write_bytes(b"")
    candidate = d / "synthetic-parent.pt"
    candidate.write_bytes(b"fixture-only-not-a-model")
    event_file = d / "events.jsonl"
    event_file.write_bytes(b"".join(canonical(e) + b"\n" for e in events))
    chunk_refs = {p.name: ref(p) for p in d.glob("search-aliases-*.bin")}
    helper = dict(directory=str(d), model_sha256=sha(feature), prior_sha256=sha(prior))
    sources = {n: sha(producer / n) for n in ("collector.py", "run_collection.py")}
    reg = dict(
        schema="own-nnue-ownq-collection-registration-v2",
        status="registered",
        seed=20262905,
        original_first_epoch=1,
        original_deadline_epoch=2,
        operator_end_epoch=100,
        parent_candidate=ref(candidate),
        root_pool=write(d / "pool.json", pool),
        protected_aliases=ref(protected),
        search_helper=ref(producer / "collector.py"),
        producer_source_sha256=sources,
        parent_helpers=helper,
    )
    reg["root_pool"].update(
        selection_path=selection_ref["path"],
        selection_sha256=selection_ref["sha256"],
        teacher_labels_path=labels_ref["path"],
        teacher_labels_sha256=labels_ref["sha256"],
    )
    rr = write(d / "registration.json", reg)
    receipt = dict(
        schema="own-nnue-ownq-collection-receipt-v2",
        status="PASS-exact-row-budget",
        train_rows=1024,
        registration_sha256=rr["sha256"],
        parent_candidate_sha256=sha(candidate),
        teacher_labels_used=False,
        search=dict(nodes=8192, qdepth=2, max_depth=8),
        root_pool_sha256=reg["root_pool"]["sha256"],
        protected_aliases_sha256=sha(protected),
        original_deadline_epoch=2,
        finished_epoch=1.5,
        operator_end_epoch=100,
        producer_source_sha256=sources,
        parent_helpers=helper,
        teacher_selection_path=selection_ref["path"],
        teacher_selection_sha256=selection_ref["sha256"],
        teacher_labels_path=labels_ref["path"],
        teacher_labels_sha256=labels_ref["sha256"],
        events_sha256=sha(event_file),
        events_bytes=event_file.stat().st_size,
        alias_chunks=[
            dict(file=n, bytes=Path(r["path"]).stat().st_size, sha256=r["sha256"])
            for n, r in chunk_refs.items()
        ],
        selected_root_order_sha256=result["selected_root_order_sha256"],
        games=result["games"],
        starts_considered=result["starts_considered"],
        exposed_board_alias_count=len(result["exposed_piece_aliases"]),
        training_row_ids=result["training_row_ids"],
        all_actor_rows=len(result["rows"]),
        periodic_independent_search_rows=[
            result["training_row_ids"][i] for i in (0, 204, 409, 614, 819, 1023)
        ],
    )
    seal = dict(
        schema="NNUE-own1024-dataset-conversion-seal-v2",
        registration=rr,
        receipt=write(d / "receipt.json", receipt),
        producer_directory=str(producer),
        events=ref(event_file),
        alias_chunks=chunk_refs,
        features=ref(feature),
        prior=ref(prior),
    )
    return seal, d, result


def test_full1024_synthetic_complete_ledger_no_terminal_fabrication(ledger):
    seal, _, result = ledger
    before = copy.deepcopy(seal)
    data, provenance = convert(seal)
    ds, p = json.loads(data), json.loads(provenance)
    assert len(ds["rows"]) == 1024 and ds["phase"] == "own-learning"
    assert [r["target"] for r in ds["rows"]] == [0.2] * 1024
    assert p["collection_receipt_sha256"] == seal["receipt"]["sha256"]
    assert p["teacher_labels_used"] is False
    assert any(g["status"].startswith("unknown-") for g in result["games"])
    assert all(r["own_wdl_mover"] is None for r in p["trace"] if r["outcome_status"] == "UNKNOWN")
    assert seal == before


@pytest.mark.parametrize("mutation", ["source", "alias-span", "mover", "final-history"])
def test_hash_resealed_semantic_corruption_rejected(ledger, tmp_path, mutation):
    seal, _, _ = ledger
    s = copy.deepcopy(seal)
    events = [json.loads(line) for line in Path(s["events"]["path"]).read_bytes().splitlines()]
    if mutation == "source":
        s["features"]["sha256"] = "0" * 64
    else:
        if mutation == "alias-span":
            next(e for e in events if e["type"] == "search_row")["row"]["search_alias_ref"][
                "offset_bytes"
            ] = 8
        elif mutation == "mover":
            row = next(e for e in events if e["type"] == "search_row")["row"]
            row["mover"] = "black" if row["mover"] == "white" else "white"
        else:
            next(e for e in events if e["type"] == "game_end")["final_state"]["history_uci"] = []
        f = tmp_path / "events.jsonl"
        f.write_bytes(b"".join(canonical(e) + b"\n" for e in events))
        s["events"] = ref(f)
        receipt = json.loads(Path(s["receipt"]["path"]).read_bytes())
        receipt.update(events_sha256=sha(f), events_bytes=f.stat().st_size)
        s["receipt"] = write(tmp_path / "receipt.json", receipt)
    with pytest.raises(ValueError):
        convert(s)


def test_new_clock_is_explicit_without_old_cap_or_reset():
    clock(dict(mode="proof", first=1791330000, deadline=1791330600, operator_end_epoch=1791331000))
    with pytest.raises(ValueError):
        clock(
            dict(mode="proof", first=1791330000, deadline=1791330601, operator_end_epoch=1791331000)
        )
    with pytest.raises(ValueError):
        clock(dict(mode="fresh-fit", first=1, deadline=1802, operator_end_epoch=3000))
