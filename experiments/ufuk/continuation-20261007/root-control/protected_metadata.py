"""Reconstruct existing split exclusions only; no search, inference or labels."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import time
import chess

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def alias(board):
    placement = min(board.board_fen(), board.mirror().board_fen())
    return int.from_bytes(hashlib.sha256(placement.encode()).digest()[:8], 'little', signed=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    reg = json.loads(args.registration.read_bytes())
    def guard():
        assert reg['first_epoch'] <= time.time() < reg['deadline_epoch'] <= reg['operator_end_epoch']
    guard()
    old = json.loads(Path(reg['teacher_registration']).read_bytes())
    producer_path = Path(reg['producer_path'])
    assert sha(producer_path) == old['source_sha256'][str(producer_path)]
    sys.path.insert(0, str(producer_path.parent))
    spec = importlib.util.spec_from_file_location('frozen_teacher_metadata', producer_path)
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    groups, positions, histories, receipt = producer.reconstruct(old, guard)
    guard()
    aliases = sorted({alias(chess.Board(fen4+' 0 1')) for fen4 in positions})
    output = Path(reg['output'])
    output.mkdir(parents=True, exist_ok=False)
    binary = output/'protected-aliases.bin'
    binary.write_bytes(struct.pack(f'<{len(aliases)}q', *aliases))
    (output/'excluded-fullhistories.json').write_text(json.dumps(sorted(histories))+'\n')
    result = dict(status='PASS-metadata-only-not-strength', seed=old['seed'],
        receipt=receipt, training_trajectory_count=len(groups),
        protected_alias_count=len(aliases), protected_alias_sha256=sha(binary),
        teacher_registration_sha256=sha(reg['teacher_registration']),
        registration_sha256=sha(args.registration), finished_epoch=time.time(),
        alias_definition='min(piece placement, vertical-color mirror); SHA256 first8 signed little64',
        stockfish_calls=0, model_calls=0, SGD_updates=0)
    guard()
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
