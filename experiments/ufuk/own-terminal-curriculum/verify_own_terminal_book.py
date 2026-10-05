"""Read-only verification of loader compatibility, mixture counts and own-prefix provenance."""
import argparse
import json
from collections import Counter
from pathlib import Path

import chess
from own_terminal_book import DISTANCES, build, sha

from harbichess.training.torch_online_learner import read_online_train_book


def verify(manifest, book):
    manifest_before, book_before = sha(manifest), sha(book)
    spec = json.loads(Path(manifest).read_text())
    packet = json.loads(Path(book).read_text())
    if packet['source_manifest_sha256'] != sha(manifest):
        raise ValueError('source manifest identity mismatch')
    original = Path(spec['original_train_book']['path'])
    for row in [spec['original_train_book'], *spec['journals']]:
        if sha(row['path']) != row['sha256']:
            raise ValueError('source SHA mismatch')
    expected, games = build(original, [x['path'] for x in spec['journals']], seed=spec['seed'])
    for key, value in expected.items():
        if packet[key] != value:
            raise ValueError('frozen selection differs: ' + key)
    openings = read_online_train_book(Path(book))
    original_rows = json.loads(original.read_text())['splits']['train']
    origins = {x['source_game'] for x in original_rows}
    kinds = Counter(x['curriculum_provenance']['kind'] for x in packet['splits']['train'])
    if kinds != {'original': 3 * len(original_rows),
                 'own-completed-prefix': len(original_rows)}:
        raise ValueError('mixture is not exactly75/25')
    lookup = {(x['journal'], x['game_index']): x for x in games}
    for row, opening in zip(packet['splits']['train'], openings, strict=True):
        provenance = row['curriculum_provenance']
        source = provenance.get('origin_source_id', provenance.get('source'))
        if source not in origins:
            raise ValueError('alias hides non-TRAIN underlying source')
        if provenance['kind'] == 'original':
            continue
        game = lookup[(provenance['journal'], provenance['game_index'])]
        distance = provenance['distance']
        if distance not in DISTANCES or game['fresh_plies'] < 32:
            raise ValueError('invalid backward distance/fresh history')
        terminal = game['terminal']
        if opening.state.moves != terminal.moves[:-distance]:
            raise ValueError('curriculum root is not own terminal exact history prefix')
        # Independent python-chess oracle: no cached-rule machinery or source-result target.
        board = chess.Board(opening.state.root_fen)
        for move in opening.state.moves:
            parsed = chess.Move.from_uci(move.uci)
            if parsed not in board.legal_moves:
                raise ValueError('illegal fullhistory prefix')
            board.push(parsed)
        if board.outcome(claim_draw=True) is not None:
            raise ValueError('terminal curriculum start')
        if board.fen() != row['opening']['fen']:
            raise ValueError('prefix FEN mismatch')
    if sha(manifest) != manifest_before or sha(book) != book_before:
        raise ValueError('manifest/book changed during verification')
    for row in [spec['original_train_book'], *spec['journals']]:
        if sha(row['path']) != row['sha256']:
            raise ValueError('source changed during verification')
    return {'status': 'pass-frozen-own-TRAIN-prefix-mixture-no-prior-labels',
            'book_sha256': sha(book), 'source_manifest_sha256': sha(manifest),
            'rows': len(openings), 'kinds': dict(kinds), 'eligible_completed_games': len(games),
            'runtime_curriculum_probability': 0.25,
            'curriculum_sampling': 'uniform factory draws; empirical frozen runtime pool',
            'fullhistory_prefixes_independently_replayed': kinds['own-completed-prefix']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('book', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    result = verify(args.manifest, args.book)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
