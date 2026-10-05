"""Bind frozen blind TRAIN replay evidence before sampling any formal8 clock."""

import json
from pathlib import Path

MAIN_SHA = "b56a48bc75815c5a91dc0ff3f5749c46bc683ea3046437611909ca625e4d549d"
TINY_SHA = "1def37c9608dd938de52adef9b4327da85bd64497f54d374ea7884c55a251ef7"
BOOKS = {
    "6d860c4c9a08914a50c6dfe59623f32755cd5185ccc68e3a52645f41efbd6252",
    "a2885098be18e8a9e45a5ac003f89a504e50877074c80fe809f01f23443aa9ff",
}


def validate(binding, sha):
    assert set(binding) == {"main", "tiny"}
    values = {}
    for name, expected in [("main", MAIN_SHA), ("tiny", TINY_SHA)]:
        item = binding[name]
        assert item["sha256"] == expected and sha(item["path"]) == expected
        values[name] = json.loads(Path(item["path"]).read_text())
    return validate_documents(values["main"], values["tiny"])


def validate_documents(main, tiny):
    assert (
        main["schema"]
        == "family8-96-candidate-root-vs-all-observed-own-training-position-key-audit-v1"
    )
    assert main["status"] == "pass-zero-overlap"
    assert main["candidate_roots"] == 96 and main["matched_roots_count"] == 0
    assert main["total_closed_epoch_rows"] == 1179648 and main["journal_count"] == 36
    assert {b["sha256"] for b in main["candidate_books"]} == BOOKS
    assert tiny["schema"] == "family8-tiny-source7-source8-train-prefix-overlap-supplement-v1"
    assert tiny["status"] == "pass-zero-overlap"
    assert tiny["candidate_roots"] == 96 and tiny["matched_roots_count"] == 0
    assert tiny["matched_root_keys"] == [] and tiny["main_overlap_receipt_sha256"] == MAIN_SHA
    assert tiny["total_unique_rows_legally_replayed"] == 4096
    assert tiny["fullhistory_prepost_position_keys_examined"] == 8192
    assert tiny["source7_unique_rows_legally_replayed"] == 2048
    assert tiny["source8_closed_epochs"] == [1, 2] and len(tiny["journals"]) == 4
    assert {b["sha256"] for b in tiny["candidate_books"]} == BOOKS
    return {
        "main": MAIN_SHA,
        "tiny": TINY_SHA,
        "scope": "observed TRAIN replay only; no heldout game outcomes",
    }
