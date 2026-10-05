"""Frozen own-terminal start curriculum; source outcomes are never learner labels."""
import argparse
import copy
import gzip
import hashlib
import json
import random
from pathlib import Path

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState
from harbichess.training.torch_online_learner import read_online_train_book

DISTANCES = (2, 4, 8, 16, 32)
E8 = 'e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03'
SOURCE = '2312652dc52a894e9726f48321117cf114270355'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(1024 * 1024):
            h.update(data)
    return h.hexdigest()


def state(value):
    return ChessState(value['root_fen'], tuple(ChessMove(x) for x in value['moves']))


def actions(path):
    """Parse one action at a time; retain no large policy arrays after each row."""
    decoder = json.JSONDecoder()
    with gzip.open(path, 'rt') as stream:
        buffer = stream.read(65536)
        marker = '"actions":['
        while marker not in buffer:
            chunk = stream.read(65536)
            if not chunk or len(buffer) > 1024 * 1024:
                raise ValueError('journal action array missing')
            buffer += chunk
        buffer = buffer.split(marker, 1)[1]
        while True:
            buffer = buffer.lstrip(' \n\r\t,')
            if buffer.startswith(']'):
                # Consume through gzip EOF to validate the compressed footer;
                # the immutable source receipt binds the complete journal bytes.
                tail = buffer[-64:]
                while chunk := stream.read(65536):
                    tail = chunk[-64:]
                if not tail.rstrip().endswith('}'):
                    raise ValueError('incomplete canonical journal ending')
                return
            try:
                row, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = stream.read(65536)
                if not chunk:
                    raise ValueError('truncated journal action') from None
                buffer += chunk
                continue
            yield row['transition']
            buffer = buffer[end:]


def completed_games(journals, original):
    """Verify completed fresh games and select length/rules eligible games only."""
    rules = PythonChessRules(board_cache_size=8192)
    roots = {row.source_id: row.state for row in read_online_train_book(original)}
    completed, counts = [], {'completed': 0, 'short': 0, 'terminal_prefix': 0, 'unknown': 0}
    for journal in journals:
        active = {}
        for row in actions(journal):
            key = row['game_index']
            pre, post = state(row['pre']), state(row['post'])
            source = row['source_id']
            if source not in roots:
                raise ValueError('journal source is outside original TRAIN opening book')
            if key not in active:
                if pre != roots[source]:
                    raise ValueError('game begins outside its complete fresh opening history')
                active[key] = {'start': pre, 'last': pre, 'source': source, 'plies': 0}
            game = active[key]
            if pre != game['last'] or source != game['source']:
                raise ValueError('discontinuous own game history')
            if post != ChessState(pre.root_fen, (*pre.moves, ChessMove(row['action']))):
                raise ValueError('own post state does not match legal action')
            game['last'] = post
            game['plies'] += 1
            observed = row['terminal_result']
            if observed is not None:
                actual = rules.outcome(post, claim_draw=True)
                if actual is None or actual.result.value != observed:
                    raise ValueError('own terminal marker disagrees with exact rules')
                counts['completed'] += 1
                if game['plies'] < max(DISTANCES):
                    counts['short'] += 1
                else:
                    prefixes = [ChessState(post.root_fen, post.moves[:-distance])
                                for distance in DISTANCES]
                    if any(rules.outcome(s, claim_draw=True) is not None for s in prefixes):
                        counts['terminal_prefix'] += 1
                    else:
                        completed.append({'journal': str(Path(journal).resolve()),
                                          'game_index': key, 'source': source,
                                          'fresh_plies': game['plies'], 'terminal': post,
                                          'observed_result_provenance_only': observed})
                del active[key]
            elif row['rollout_cutoff']:
                counts['unknown'] += 1
                del active[key]
        counts['unknown'] += len(active)
    if not completed:
        raise ValueError('no own completed games eligible for all five distances')
    return completed, counts


def build(original, journals, *, seed):
    games, counts = completed_games(journals, original)
    original_book = json.loads(Path(original).read_text())
    rng = random.Random(seed)
    rows = []
    for replica in range(3):
        for index, row in enumerate(original_book['splits']['train']):
            new = copy.deepcopy(row)
            new['source_game'] = f'ownmix:original:{replica}:{index}'
            new['curriculum_provenance'] = {
                'kind': 'original', 'origin_source_id': row['source_game']}
            rows.append(new)
    rules = PythonChessRules()
    for index in range(len(original_book['splits']['train'])):
        game = games[rng.randrange(len(games))]
        distance = DISTANCES[rng.randrange(len(DISTANCES))]
        terminal = game['terminal']
        prefix = ChessState(terminal.root_fen, terminal.moves[:-distance])
        rows.append({'source_game': f'ownmix:curriculum:{index}', 'root_ply': prefix.ply,
                     'opening': {'root_fen': prefix.root_fen,
                                 'moves': [m.uci for m in prefix.moves],
                                 'fen': rules.view(prefix).fen},
                     'curriculum_provenance': {'kind': 'own-completed-prefix',
                         **{k: v for k, v in game.items() if k != 'terminal'},
                         'distance': distance, 'terminal_ply': terminal.ply,
                         'terminal_fullhistory_sha256': hashlib.sha256(json.dumps(
                             [terminal.root_fen, [m.uci for m in terminal.moves]],
                             separators=(',', ':')).encode()).hexdigest()}})
    return {'schema': 1,
            'source': 'Own TRAIN completed-game backward starts, original raw outcomes unused',
            'curriculum_schema': 'frozen-own-terminal-start-mix-v1', 'seed': seed,
            'original_book_sha256': sha(original),
            'journal_sha256': {str(Path(p).resolve()): sha(p) for p in journals},
            'distances': list(DISTANCES), 'curriculum_probability': 0.25,
            'eligible_own_completed_games': len(games), 'selection_counts': counts,
            'sampling': 'factory-uniform eligible game then distance; runtime-uniform frozen rows',
            'label_contract': ('Prior terminal result is provenance only; '
                               'fresh continuation outcome required'),
            'splits': {'train': rows}}, games


def freeze(manifest, output):
    manifest_before = sha(manifest)
    spec = json.loads(Path(manifest).read_text())
    if spec['source_commit'] != SOURCE or spec['initial_weights_sha256'] != E8:
        raise ValueError('requires pinned own TRAIN producer and original e8 anchor')
    original = Path(spec['original_train_book']['path'])
    journals = [Path(x['path']) for x in spec['journals']]
    if not journals or len(set(journals)) != len(journals):
        raise ValueError('unique frozen journals required')
    for row in [spec['original_train_book'], *spec['journals']]:
        if sha(row['path']) != row['sha256']:
            raise ValueError('frozen input SHA mismatch')
    if not spec.get('source_runs'):
        raise ValueError('pinned original own-run metadata required')
    covered = set()
    for row in spec['source_runs']:
        metadata = Path(row['metadata_path'])
        if sha(metadata) != row['metadata_sha256']:
            raise ValueError('own source metadata SHA mismatch')
        data = json.loads(metadata.read_text())
        if (data['source_commit'] != SOURCE
                or data['inputs']['initial_weights']['sha256'] != E8
                or data['inputs']['book']['sha256'] != sha(original)):
            raise ValueError('own source metadata producer/anchor/TRAIN book mismatch')
        for journal in journals:
            if journal.parent == metadata.parent / 'journal':
                epoch = int(journal.name.removeprefix('epoch-').removesuffix('.json.gz'))
                if not 1 <= epoch <= 40:
                    raise ValueError('outside declared own TRAIN epochs1..40')
                covered.add(journal)
    if covered != set(journals):
        raise ValueError('journal outside frozen own source runs')
    book, _ = build(original, journals, seed=spec['seed'])
    book['source_manifest_sha256'] = manifest_before
    book['source_commit'] = SOURCE
    for row in [spec['original_train_book'], *spec['journals']]:
        if sha(row['path']) != row['sha256']:
            raise ValueError('source changed during factory')
    if sha(manifest) != manifest_before:
        raise ValueError('source manifest changed during factory')
    for row in spec['source_runs']:
        if sha(row['metadata_path']) != row['metadata_sha256']:
            raise ValueError('source metadata changed during factory')
    with Path(output).open('x') as stream:
        stream.write(json.dumps(book, indent=2) + '\n')
    read_online_train_book(Path(output))
    return book


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    book = freeze(args.manifest, args.output)
    print(json.dumps({'book_sha256': sha(args.output), 'rows': len(book['splits']['train']),
                      'eligible_games': book['eligible_own_completed_games']}))


if __name__ == '__main__':
    main()
