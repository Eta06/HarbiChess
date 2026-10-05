"""Read-only full-history overlap audit for family8's frozen 96 roots.

The remote audit code is transmitted via the already-authorized Colab SSH helper;
no model, engine, or optimizer is invoked. Results are scoped to prospective
root-vs-own-training position-key overlap only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shlex
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent
BOOKS = (
    ROOT / "20260709" / "opening-splits.json",
    ROOT / "20260710" / "opening-splits.json",
)
BOOK_SHA = (
    "6d860c4c9a08914a50c6dfe59623f32755cd5185ccc68e3a52645f41efbd6252",
    "a2885098be18e8a9e45a5ac003f89a504e50877074c80fe809f01f23443aa9ff",
)
SOURCE7_CLI_JOURNALS = {
    "00000001": "5f3b63538cb67e6d4fcdcc9e204d0993039e0d62fe7498ace14421c4ace03bc3",
    "00000002": "097e31cf396e0133d0e51ada7b7d27213a4b8dc85a082a62a7cd13ed9e55df59",
}


REMOTE = r'''import gzip, hashlib, json, os, re, sys, time
from collections import defaultdict
from pathlib import Path
import chess

STARTED = time.time()
MONO = time.monotonic()
LIMIT = MONO + 560.0
ROOT = Path('/content/harbichess-runs')
CANDIDATE = set(json.loads(Path(sys.argv[1]).read_text()))
assert len(CANDIDATE) == 96
S7_CLI_EXPECTED = {
  '00000001': '5f3b63538cb67e6d4fcdcc9e204d0993039e0d62fe7498ace14421c4ace03bc3',
  '00000002': '097e31cf396e0133d0e51ada7b7d27213a4b8dc85a082a62a7cd13ed9e55df59',
}
SPECS = [
 ('method4-20261425', ROOT/'ownsearch-method4-seed-20261425/run', range(1,11), 'closed'),
 ('method4-20261426', ROOT/'ownsearch-method4-seed-20261426/run', range(1,10), 'closed'),
 ('method5-20261525', ROOT/'search-acting-method5-seed-20261525/run', range(1,9), 'closed'),
 ('method5-20261526', ROOT/'search-acting-method5-seed-20261526/run', range(1,9), 'closed'),
 ('method7-development-20261775', ROOT/'certificate-source7-fullshape900-20261005/run', range(1,2), 'development'),
]
def check_time():
    assert time.monotonic() < LIMIT, 'fixed original 600-second audit ceiling'
def digest(b): return hashlib.sha256(b).hexdigest()
def position_key(board): return ' '.join(board.fen().split()[:4])
def sig(state): return (state['root_fen'], tuple(state['moves']))
def replay(state):
    board = chess.Board(state['root_fen'])
    assert board.is_valid()
    for uci in state['moves']:
        check_time()
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, ('illegal-full-history-prefix', uci)
        board.push(move)
    return board

matches = {}
receipts = []
total_rows = total_known = total_unknown = total_position_keys = total_replayed = 0
for label, run, expected_epochs, kind in SPECS:
    check_time()
    assert run.is_dir() and not run.is_symlink(), ('missing-run', label)
    meta_path = run/'metadata.json'
    meta_bytes = meta_path.read_bytes()
    meta = json.loads(meta_bytes)
    expected_source = 'c022bc1605b44c3089439da5c6efb7bd4db4ff81' if label.startswith('method7') else None
    if expected_source:
        assert meta.get('source_commit') == expected_source
    indices = list(expected_epochs)
    if label.startswith('method5'):
        # Conservatively include any extra journal file, even if it lacks a closed native.
        present = sorted(int(p.name.split('.')[0].split('-')[-1]) for p in (run/'journal').glob('epoch-*.json.gz'))
        extras = [i for i in present if i not in indices]
        indices.extend(extras)
    else:
        extras = []
    previous = {}
    run_receipts = []
    for index in indices:
        check_time()
        journal = run/'journal'/f'epoch-{index:08d}.json.gz'
        raw = journal.read_bytes()
        record = json.loads(gzip.decompress(raw))
        assert record.get('epoch') == index
        collection = record['collection']
        actions = collection['actions']
        assert record.get('fresh_transitions') == len(actions)
        known = unknown = prepost = 0
        for ordinal, action in enumerate(actions):
            if ordinal % 256 == 0: check_time()
            trans = action['transition']
            pre, post = trans['pre'], trans['post']
            assert post['root_fen'] == pre['root_fen']
            assert post['moves'] == pre['moves'] + [trans['action']]
            slot = trans['slot']
            game_id = (slot, trans['source_id'], trans['game_index'])
            if game_id in previous:
                board, expected_pre = previous[game_id]
                assert sig(pre) == expected_pre, ('broken-actor-history-chain', label, index, ordinal)
            else:
                board = replay(pre)
                total_replayed += 1
            pre_key = position_key(board)
            prepost += 1
            total_position_keys += 1
            if pre_key in CANDIDATE:
                matches.setdefault(pre_key, []).append({
                    'source': label, 'journal_epoch': index, 'row': ordinal,
                    'where': 'pre', 'slot': slot, 'game_index': trans['game_index'],
                    'source_id': trans['source_id'],
                })
            move = chess.Move.from_uci(trans['action'])
            assert move in board.legal_moves, ('illegal-actor-transition', label, index, ordinal)
            board.push(move)
            post_key = position_key(board)
            prepost += 1
            total_position_keys += 1
            if post_key in CANDIDATE:
                matches.setdefault(post_key, []).append({
                    'source': label, 'journal_epoch': index, 'row': ordinal,
                    'where': 'post', 'slot': slot, 'game_index': trans['game_index'],
                    'source_id': trans['source_id'],
                })
            previous[game_id] = (board, sig(post))
            if trans.get('terminal_result') is None: unknown += 1
            else: known += 1
        n = len(actions)
        assert known + unknown == n
        total_rows += n; total_known += known; total_unknown += unknown
        rel = str(journal.relative_to(ROOT))
        row_receipt = {
            'path': str(journal), 'label': label, 'epoch': index,
            'classification': 'extra-observed-not-closed' if index in extras else kind,
            'compressed_bytes': len(raw), 'sha256': digest(raw),
            'record_schema': record.get('schema'),
            'header_fresh_transitions': record.get('fresh_transitions'),
            'rows': n, 'known_terminal_rows': known, 'unknown_rows': unknown,
            'fullhistory_prepost_keys': prepost,
            'fresh_game_prefixes_legally_replayed': total_replayed,
            'metadata_sha256': digest(meta_bytes),
        }
        if index in expected_epochs and (run/'checkpoints'/f'epoch-{index:08d}'/'checkpoint.json').is_file():
            row_receipt['native_checkpoint_present'] = True
        else:
            row_receipt['native_checkpoint_present'] = False
        receipts.append(row_receipt); run_receipts.append(row_receipt)

# Source7 split/whole proof: compare original public CLI receipt hashes and bytes.
tiny_base = ROOT/'certificate-source7-clean-CLI600-20261005'
tiny = []
for epoch, expected_hash in sorted(S7_CLI_EXPECTED.items()):
    check_time()
    whole = tiny_base/'whole'/'journal'/f'epoch-{epoch}.json.gz'
    split = tiny_base/'split'/'journal'/f'epoch-{epoch}.json.gz'
    wb, sb = whole.read_bytes(), split.read_bytes()
    assert digest(wb) == digest(sb) == expected_hash
    record = json.loads(gzip.decompress(wb))
    assert record['fresh_transitions'] == 1024 and len(record['collection']['actions']) == 1024
    tiny.append({'epoch': int(epoch), 'whole_sha256': digest(wb), 'split_sha256': digest(sb),
                 'byte_identical': wb == sb, 'rows': 1024,
                 'schema': record.get('schema'), 'known_terminal_rows': sum(a['transition'].get('terminal_result') is not None for a in record['collection']['actions']),
                 'unknown_rows': sum(a['transition'].get('terminal_result') is None for a in record['collection']['actions'])})

out = {
 'schema': 'family8-96-candidate-root-vs-all-observed-own-training-position-key-audit-v1',
 'status': 'pass-zero-overlap' if not matches else 'fail-training-prefix-overlap',
 'scope': 'read-only TRAIN/development full-history actor journal audit; no weights/model/engine/optimizer/outcome-quality queries',
 'started_epoch': STARTED, 'elapsed_seconds': time.monotonic()-MONO, 'fixed_ceiling_seconds': 560,
 'candidate_roots': 96, 'candidate_position_key_definition': 'python-chess full-history replay then FEN first four fields, matching consumer635413 history_openings.freeze',
 'total_closed_epoch_rows': total_rows, 'known_terminal_rows': total_known, 'unknown_rows': total_unknown,
 'pre_post_position_keys_examined': total_position_keys,
 'fresh_game_full_history_prefixes_legally_replayed': total_replayed,
 'journal_count': len(receipts), 'journal_receipts': receipts,
 'source7_tiny_cli_duplicate_proof': tiny,
 'matched_root_keys': [{'position_key': key, 'occurrences': rows} for key, rows in sorted(matches.items())],
 'matched_roots_count': len(matches),
 'method5_extra_journals_conservatively_counted': [r for r in receipts if r['classification']=='extra-observed-not-closed'],
}
target = Path(sys.argv[2])
target.write_text(json.dumps(out, sort_keys=True, separators=(',',':'))+'\n')
print(json.dumps({'status': out['status'], 'elapsed_seconds': out['elapsed_seconds'],
 'rows': total_rows, 'known': total_known, 'unknown': total_unknown,
 'prepost_keys': total_position_keys, 'replayed_game_prefixes': total_replayed,
 'journals': len(receipts), 'overlap_roots': len(matches), 'result_sha256': digest(target.read_bytes())}, sort_keys=True))
'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    sys.path.insert(0, "/workspace/work/harbichess/a100")
    import ssh_colab_access

    roots: set[str] = set()
    for path, expected_sha in zip(BOOKS, BOOK_SHA, strict=True):
        assert sha(path.read_bytes()) == expected_sha
        book = json.loads(path.read_text())
        assert book["schema"] == 1 and sorted(book["splits"]) == ["arena"]
        for record in book["splits"]["arena"]:
            fen = record["opening"]["fen"]
            roots.add(" ".join(fen.split()[:4]))
    assert len(roots) == 96
    target = ROOT / "train-prefix-overlap-audit-result.json"
    assert not target.exists()
    run_id = f"own8-overlap-{int(time.time())}"
    remote_dir = f"/tmp/{run_id}"
    key_blob = json.dumps(sorted(roots), separators=(",", ":"))
    setup = (
        "import pathlib,base64; p=pathlib.Path(" + repr(remote_dir) + "); p.mkdir(mode=0o700); "
        "(p/'candidate.json').write_bytes(base64.b64decode(" + repr(__import__('base64').b64encode(key_blob.encode()).decode()) + ")); "
        "(p/'audit.py').write_bytes(base64.b64decode(" + repr(__import__('base64').b64encode(REMOTE.encode()).decode()) + ")); "
        "print(str(p))"
    )
    setup_result = ssh_colab_access.call("python3 -c " + shlex.quote(setup), timeout=60, capture_output=True, text=True)
    if setup_result.returncode:
        raise RuntimeError(f"remote scratch setup failed ({setup_result.returncode}); no source/data changes")
    launch = (
        f"nohup env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        f"python3 {remote_dir}/audit.py {remote_dir}/candidate.json {remote_dir}/result.json "
        f">{remote_dir}/audit.log 2>&1 </dev/null & echo $!"
    )
    launched = ssh_colab_access.call(launch, timeout=60, capture_output=True, text=True)
    if launched.returncode:
        raise RuntimeError(f"remote audit launch failed ({launched.returncode})")
    pid = launched.stdout.strip().splitlines()[-1]
    (ROOT / "train-prefix-overlap-audit-launch.json").write_text(json.dumps({
        "status": "running-read-only-audit", "remote_pid": pid, "remote_scratch": remote_dir,
        "candidate_root_count": len(roots), "book_sha256": list(BOOK_SHA),
        "launched_epoch": time.time(), "absolute_local_audit_deadline_epoch": time.time() + 600,
        "remote_script_sha256": sha(REMOTE.encode()), "no_jobs_models_or_engines": True,
    }, indent=2) + "\n")
    print(json.dumps({"status": "started", "pid": pid, "remote_scratch": remote_dir,
                      "candidate_roots": len(roots), "script_sha256": sha(REMOTE.encode())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
