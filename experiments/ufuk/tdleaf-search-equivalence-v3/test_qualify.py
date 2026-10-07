import gzip
import hashlib
import random

import chess

import qualify
import search_original
import search_pv


def test_pv_instrumentation_preserves_search_and_exact_ordered_evaluator_inputs():
    board = chess.Board()
    baseline_trace = qualify.Trace(lambda b: (b.piece_type_at(chess.E4) or 0) / 6 - 0.2)
    pv_trace = qualify.Trace(lambda b: (b.piece_type_at(chess.E4) or 0) / 6 - 0.2)
    def traced_baseline_eval(position):
        baseline_trace.observe(position)
        return baseline_trace(position)

    base = search_original.BudgetSearch(
        traced_baseline_eval, nodes=512, quiescence_plies=2, max_depth=4
    ).search(board)
    pv = search_pv.BudgetSearch(
        pv_trace,
        input_observer=pv_trace.observe,
        nodes=512,
        quiescence_plies=2,
        max_depth=4,
    ).search(board)
    assert qualify.result_payload(base) == qualify.result_payload(pv)
    assert baseline_trace.aliases == pv_trace.observed_aliases
    assert baseline_trace.aliases == pv_trace.aliases
    assert baseline_trace.evaluated_inputs == pv_trace.evaluated_inputs
    assert baseline_trace.observed_aliases == baseline_trace.aliases
    assert baseline_trace.observed_inputs == [
        {"fen": row["fen"], "history_uci": row["history_uci"]}
        for row in baseline_trace.evaluated_inputs
    ]
    assert pv_trace.observed_inputs == [
        {"fen": row["fen"], "history_uci": row["history_uci"]} for row in pv_trace.evaluated_inputs
    ]
    leaf_board = qualify.verify_pv(board, pv, pv_trace.observed_aliases)
    assert len(pv.pv) == pv.leaf.ply
    assert leaf_board.is_valid()


def test_root_selection_is_fixed_order_and_rejects_protected_actual_root():
    rng = random.Random(42)
    rows = []
    histories = set()
    while len(rows) < 4096:
        board = chess.Board()
        for _ in range(rng.randint(3, 18)):
            if board.outcome(claim_draw=True) is not None:
                break
            board.push(rng.choice(list(board.legal_moves)))
        history = [move.uci() for move in board.move_stack]
        key = tuple(history)
        if len(history) < 2 or key in histories or board.outcome(claim_draw=True) is not None:
            continue
        histories.add(key)
        rows.append(
            {
                "role": "TRAIN",
                "root_fen": chess.STARTING_FEN,
                "prefix_uci": history,
                "root_id": f"root-{len(rows)}",
            }
        )
    pool = {"rows": rows}
    start_alias = qualify.alias(chess.Board())
    selected, considered = qualify.choose_roots(pool, {start_alias})
    assert len(selected) == 24
    assert considered == 24
    assert all(qualify.alias(qualify.replay(row)) != start_alias for row in selected)


def _packet_fixture():
    query = {"fen": chess.Board().fen(), "history_uci": [], "value_hex": "-0x0.0p+0"}
    result = {"move": "e2e4", "value": -0.0, "value_hex": "-0x0.0p+0"}
    packet = {
        "schema": qualify.PACKET_SCHEMA,
        "index": 0,
        "root_id": "root-0",
        "baseline_result": result,
        "pv_result": result,
        "baseline_ordered_query_packets": [query],
        "baseline_observed_input_histories": [{"fen": query["fen"], "history_uci": []}],
        "baseline_observed_aliases": [11],
        "pv_observed_input_histories": [{"fen": query["fen"], "history_uci": []}],
        "pv_ordered_query_packets": [query],
        "baseline_ordered_aliases": [11],
        "pv_observed_aliases": [11],
        "pv_evaluator_call_aliases": [11],
        "baseline_protected_alias_hits": [],
        "pv_protected_alias_hits": [],
        "pv_observer_protected_alias_hits": [],
        "current_root_to_pv_aliases": [22],
        "current_root_to_pv_protected_hits": [],
    }
    summary = {
        "index": 0,
        "root_id": "root-0",
        "baseline": result,
        "pv": result,
        "protected_current_root_to_pv_path": [],
    }
    return packet, summary


def test_query_packet_roundtrip_preserves_full_histories_and_signed_zero(tmp_path):
    packet, summary = _packet_fixture()
    packed = qualify.packet_bytes(packet)
    decoded = qualify.decode_packet_bytes(packed)
    assert qualify.packet_matches_summary(decoded, summary)
    assert decoded["baseline_result"]["value_hex"] == "-0x0.0p+0"
    packet_dir = tmp_path / "packets"
    packet_dir.mkdir()
    path = packet_dir / "root-00.json.gz"
    path.write_bytes(packed)
    ref = {"path": str(path), "bytes": len(packed), "sha256": hashlib.sha256(packed).hexdigest()}
    assert qualify.read_packet_ref(packet_dir, ref, 0) == packet


def test_query_packet_rejects_truncation_trailing_bytes_and_concatenated_gzip():
    packet, _ = _packet_fixture()
    packed = qualify.packet_bytes(packet)
    for bad in (packed[:-1], packed + b"x", packed + gzip.compress(b"{}", mtime=0)):
        try:
            qualify.decode_packet_bytes(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt/trailing compressed packet was accepted")


def test_query_packet_enforces_uncompressed_per_root_bound(monkeypatch):
    packet, _ = _packet_fixture()
    packed = qualify.packet_bytes(packet)
    monkeypatch.setattr(qualify, "MAX_PACKET_RAW_BYTES", 128)
    try:
        qualify.packet_bytes(packet)
    except ValueError:
        pass
    else:
        raise AssertionError("oversized full-history packet was accepted")
    try:
        qualify.decode_packet_bytes(packed)
    except ValueError:
        pass
    else:
        raise AssertionError("over-bound decompressed query packet was accepted")


def test_query_packet_compressed_per_root_bound(monkeypatch):
    packet, _ = _packet_fixture()
    monkeypatch.setattr(qualify, "MAX_PACKET_BYTES", 16)
    try:
        qualify.packet_bytes(packet)
    except ValueError:
        pass
    else:
        raise AssertionError("oversized compressed packet was accepted")


def test_query_packet_ref_rejects_resealed_path_or_tampered_bytes(tmp_path):
    packet, _ = _packet_fixture()
    packet_dir = tmp_path / "packets"
    packet_dir.mkdir()
    path = packet_dir / "root-00.json.gz"
    packed = qualify.packet_bytes(packet)
    path.write_bytes(packed)
    ref = {"path": str(path), "bytes": len(packed), "sha256": hashlib.sha256(packed).hexdigest()}
    with path.open("ab") as stream:
        stream.write(b"tamper")
    try:
        qualify.read_packet_ref(packet_dir, ref, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered query packet was accepted")


def test_protected_path_or_unqueried_observer_input_blocks_packet():
    packet, _ = _packet_fixture()
    assert qualify.packet_protection_clear(packet, set())
    assert not qualify.packet_protection_clear(packet, {22})
    packet["pv_observer_protected_alias_hits"] = [11]
    assert not qualify.packet_protection_clear(packet, set())


def test_packet_summary_requires_exact_value_hex_and_query_history():
    packet, summary = _packet_fixture()
    assert qualify.packet_matches_summary(packet, summary)
    packet["pv_result"] = {**packet["pv_result"], "value_hex": "0x0.0p+0"}
    assert not qualify.packet_matches_summary(packet, summary)
    packet, summary = _packet_fixture()
    packet["pv_observed_input_histories"] = [{"fen": "corrupt", "history_uci": []}]
    assert not qualify.packet_matches_summary(packet, summary)
