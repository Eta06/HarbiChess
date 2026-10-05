"""Export all observed own TRAIN prefixes, including UNKNOWN tails, for new-book exclusion."""
import argparse
import hashlib
import json
from pathlib import Path

from own_terminal_book import actions, sha, state

from harbichess.chess.rules import PythonChessRules


def export(manifest, output):
    manifest_before = sha(manifest)
    spec = json.loads(Path(manifest).read_text())
    original = spec['original_train_book']
    if sha(original['path']) != original['sha256']:
        raise ValueError('original TRAIN book changed')
    rows = json.loads(Path(original['path']).read_text())['splits']['train']
    source_ids = {x['source_game'] for x in rows}
    rules = PythonChessRules(board_cache_size=8192)
    keys, histories = set(), set()
    count = 0
    for source in spec['journals']:
        if sha(source['path']) != source['sha256']:
            raise ValueError('source journal changed')
        for transition in actions(source['path']):
            if transition['source_id'] not in source_ids:
                raise ValueError('underlying non-TRAIN source')
            for name in ('pre', 'post'):
                position = state(transition[name])
                keys.add(' '.join(rules.view(position).fen.split()[:4]))
                histories.add(hashlib.sha256(json.dumps(transition[name],
                              sort_keys=True, separators=(',', ':')).encode()).hexdigest())
            count += 1
        if sha(source['path']) != source['sha256']:
            raise ValueError('source journal changed during exclusion export')
    if sha(manifest) != manifest_before or sha(original['path']) != original['sha256']:
        raise ValueError('source changed during exclusion export')
    result = {'schema': 'own-terminal-curriculum-training-exclusions-v1',
              'source_manifest_sha256': sha(manifest),
              'underlying_original_TRAIN_source_ids': sorted(source_ids),
              'all_observed_own_prefix_position_keys': sorted(keys),
              'all_observed_own_fullhistory_sha256': sorted(histories),
              'fresh_transitions_examined': count,
              'scope': 'All pre/post own TRAIN states incl UNKNOWN; no outcome filtering or scores',
              'rule': ('Fresh strength books must exclude union with '
                       'ALL earlier registered exclusions')}
    with Path(output).open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    result = export(args.manifest, args.output)
    print(json.dumps({'sha256': sha(args.output),
                      'position_keys': len(result['all_observed_own_prefix_position_keys']),
                      'underlying_sources': len(result['underlying_original_TRAIN_source_ids'])}))


if __name__ == '__main__':
    main()
