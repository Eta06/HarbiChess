"""Actual v2 string-ID wire format; no model/inference/search calls."""
import copy

import pytest
from audit_collection_six_v2 import ORDINALS, selected_rows


def packet():
    rows = [dict(root_id=f"source-trajectory:{i // 16}", local_ply=i % 16) for i in range(1024)]
    ids = [r["root_id"] + ":" + str(r["local_ply"]) for r in rows]
    return rows, ids, [ids[i] for i in ORDINALS]


def test_actual_string_identifiers_bind_prescribed_chronological_rows():
    rows, ids, selected = packet()
    assert selected_rows(rows, ids, selected) == [(ids[i], rows[i]) for i in ORDINALS]
    # Protected/ineligible raw rows may exist without changing eligible ID order.
    rows.insert(2, dict(root_id="excluded-trajectory", local_ply=0))
    assert [key for key, _row in selected_rows(rows, ids, selected)] == selected


def test_duplicate_missing_order_and_numeric_ids_rejected_before_model_load():
    rows, ids, selected = packet()
    for corrupt in [[*rows, copy.deepcopy(rows[0])], rows[1:], list(reversed(rows))]:
        with pytest.raises(ValueError):
            selected_rows(corrupt, ids, selected)
    with pytest.raises(ValueError):
        selected_rows(rows, list(range(1024)), list(ORDINALS))
    with pytest.raises(ValueError):
        selected_rows(rows, ids, list(reversed(selected)))
