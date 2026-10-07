"""Pure packet/trace tamper tests; no actual evaluator or search calls."""
import copy
import json
import struct
import subprocess
import sys
from pathlib import Path

import chess
import pytest
from audit_collection_six import ORDINALS, alias_segment, check_packet, sha


def fixture():
    b = chess.Board()
    r = dict(selected_best_uci="e2e4", raw_q_mover=0.25, nodes=100, evaluations=70,
             completed_depth=0, root_actions=20, actual_eval_calls=70,
             clipped_q_mover=0.25, mate_range_score_returned=False,
             fen4=" ".join(b.fen().split()[:4]), mover="white")
    a = {k: r[k] for k in ("selected_best_uci", "nodes", "evaluations", "completed_depth",
                           "root_actions", "actual_eval_calls")}
    a["value_hex"] = float(r["raw_q_mover"]).hex()
    return b, r, a


def test_actual_packet_comparison_rejects_score_mover_counter_and_history_tamper():
    b, r, a = fixture()
    check_packet(r, a, [-2, 1], b)
    for key, value in [("raw_q_mover", 0.3), ("mover", "black"),
                       ("nodes", 101), ("actual_eval_calls", 71), ("fen4", "wrong")]:
        corrupt = copy.deepcopy(r)
        corrupt[key] = value
        with pytest.raises(ValueError):
            check_packet(corrupt, a, [-2, 1], b)
    with pytest.raises(ValueError):
        check_packet(r, a, [1, -2], b)


def test_exact_alias_span_bounds_sha_and_mutation(tmp_path):
    p = tmp_path / "aliases.bin"
    p.write_bytes(struct.pack("<4q", -10, -2, 1, 10))
    receipt = {"alias_chunks": [dict(file=p.name, bytes=32, sha256=sha(p))]}
    row = {"search_alias_ref": dict(file=p.name, offset_bytes=8, count=2)}
    assert alias_segment(row, receipt, tmp_path) == [-2, 1]
    for offset, count in [(1, 2), (24, 2), (-8, 1), (0, True)]:
        row["search_alias_ref"].update(offset_bytes=offset, count=count)
        with pytest.raises(ValueError):
            alias_segment(row, receipt, tmp_path)
    row["search_alias_ref"].update(offset_bytes=8, count=2)
    p.write_bytes(struct.pack("<4q", -10, -2, 2, 10))
    with pytest.raises(ValueError):
        alias_segment(row, receipt, tmp_path)


def test_six_fixed_ordinals_and_cli_help_only():
    assert ORDINALS == (0, 204, 409, 614, 819, 1023)
    p = subprocess.run([sys.executable, str(Path(__file__).with_name("audit_collection_six.py")),
                        "--help"], capture_output=True, text=True, check=True)
    assert "--clock" in p.stdout and "--registration" in p.stdout
    json.dumps(dict(ordinals=ORDINALS))
