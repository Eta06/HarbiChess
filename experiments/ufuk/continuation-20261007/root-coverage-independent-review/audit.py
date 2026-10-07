"""Read-only full-history root coverage; no models, evaluations or new selection."""
import collections
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import chess

BASE = Path('/workspace/work/harbichess/continuation-20261007')
OUT = BASE / 'root-coverage-independent-review'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replay(row, key='prefix_uci'):
    board = chess.Board(row['root_fen'])
    for u in row[key]:
        assert board.outcome(claim_draw=True) is None
        move = chess.Move.from_uci(u)
        assert board.is_legal(move)
        board.push(move)
    assert board.is_valid()
    return board


def alias(board):
    raw = min(board.board_fen(), board.mirror().board_fen()).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little', signed=True)


def summary(rows):
    counters = {k: collections.Counter() for k in ['ply', 'pieces', 'material_cp', 'mover', 'queens']}
    placements, fen4, rules, histories, trajectories = [], [], [], [], []
    for row in rows:
        b = replay(row)
        pcs = len(b.piece_map())
        material = sum({1:100, 2:320, 3:330, 4:500, 5:900, 6:0}[p.piece_type]
                       for p in b.piece_map().values())
        for k, val in [('ply', b.ply()), ('pieces', pcs), ('material_cp', material),
                       ('mover', 'white' if b.turn else 'black'),
                       ('queens', len(b.pieces(chess.QUEEN, True) | b.pieces(chess.QUEEN, False)))]:
            counters[k][str(val)] += 1
        placements.append(b.board_fen())
        fen4.append(' '.join(b.fen().split()[:4]))
        rules.append(b.fen())
        histories.append(json.dumps([row['root_fen'], row['prefix_uci']], separators=(',', ':')))
        trajectories.append(row.get('trajectory_id', row['root_id'].split(':')[0]))
    return dict(rows=len(rows), histograms={k:dict(v) for k,v in counters.items()},
                opening_ply_le20=sum(v for k,v in counters['ply'].items() if int(k)<=20),
                low_material_pieces_le10=sum(v for k,v in counters['pieces'].items() if int(k)<=10),
                no_queens=counters['queens']['0'],
                unique_placements=len(set(placements)), placement_duplicates=len(rows)-len(set(placements)),
                unique_fen4=len(set(fen4)), unique_full_fen=len(set(rules)),
                unique_full_histories=len(set(histories)),
                full_history_duplicates=len(rows)-len(set(histories)),
                unique_source_trajectories=len(set(trajectories)))


def closed_collection(directory, registration):
    rp = directory / 'receipt.json'
    if not rp.is_file():
        return dict(status='LIVE-or-INCOMPLETE-no-closed-receipt', outcomes_read=False)
    receipt = json.loads(rp.read_bytes())
    assert receipt['status'] == 'PASS-exact-row-budget'
    ep = directory / 'events.jsonl'
    assert sha(ep)==receipt['events_sha256']
    events = [json.loads(x) for x in ep.read_bytes().splitlines()]
    starts, rows, ends = {}, collections.defaultdict(list), {}
    statuses, labels, wdl = collections.Counter(), collections.Counter(), collections.Counter()
    for e in events:
        if e['type']=='game_start':
            assert e['root_id'] not in starts
            starts[e['root_id']]=e
        elif e['type']=='search_row':
            r=e['row']; b=replay(r,'history_uci')
            assert ('white' if b.turn else 'black') == r['mover']
            assert ' '.join(b.fen().split()[:4]) == r['fen4']
            assert b.is_legal(chess.Move.from_uci(r['selected_best_uci']))
            rows[r['root_id']].append(r)
        elif e['type']=='game_end':
            assert e['root_id'] not in ends
            ends[e['root_id']]=e
            statuses[e['status']]+=1
            for lab in e['row_labels']: labels[str(lab)]+=1
            if e['status']=='completed-own-terminal':
                r=rows[e['root_id']][-1]; b=replay(r,'history_uci')
                assert r['selected_action_played']
                b.push_uci(r['selected_best_uci'])
                outcome=b.outcome(claim_draw=True); assert outcome is not None
                wdl['draw' if outcome.winner is None else 'white' if outcome.winner else 'black']+=1
                for rr, lab in zip(rows[e['root_id']],e['row_labels'],strict=True):
                    expected=0 if outcome.winner is None else 1 if outcome.winner == (rr['mover']=='white') else -1
                    assert lab==expected
            else:
                assert all(x is None for x in e['row_labels'])
    assert set(starts)==set(ends)
    source=[dict(root_fen=x['root_fen'],prefix_uci=x['root_prefix_uci'],root_id=x['root_id'])
            for x in starts.values()]
    return dict(status='PASS-closed-metadata-fullhistory-mover-terminal',
                outcomes_read=True,receipt_sha256=sha(rp),events_sha256=sha(ep),
                started_roots=summary(source),all_actor_rows=sum(map(len,rows.values())),
                eligible_training_rows=receipt['train_rows'],episode_statuses=dict(statuses),
                terminal_white_draw_black=dict(wdl),row_label_counts=dict(labels),
                receipt_root_order_sha256=receipt['selected_root_order_sha256'])


def main():
    os.nice(15)
    first=time.time(); before={}; reports=[]
    q=json.loads((BASE/'variant-known160-v1/protocol.json').read_bytes())
    for seed in [20262905,20262906]:
        regpath=Path(q['children'][str(seed)]['collection_registration']['path'])
        reg=json.loads(regpath.read_bytes()); poolpath=Path(reg['root_pool']['path'])
        protected=Path(reg['protected_aliases']['path'])
        assert sha(poolpath)==reg['root_pool']['sha256'] and sha(protected)==reg['protected_aliases']['sha256']
        pool=json.loads(poolpath.read_bytes()); assert len(pool['rows'])==4096
        raw=protected.read_bytes(); blocked=set(struct.unpack(f'<{len(raw)//8}q',raw))
        usable=[r for r in pool['rows'] if alias(replay(r)) not in blocked]
        chosen=sorted(usable,key=lambda r:hashlib.sha256(
            f"own-nnue-ownq-start-v1|{seed}|{r['root_id']}".encode()).digest())[:128]
        order=hashlib.sha256('\n'.join(r['root_id'] for r in chosen).encode()).hexdigest()
        own=closed_collection(Path(reg['output_path']),reg)
        assert own['receipt_root_order_sha256']==order
        mregpath=BASE/f'closedterminal-collection-{seed}/registration.json'
        mreg=json.loads(mregpath.read_bytes())
        assert mreg['root_pool']['sha256']==reg['root_pool']['sha256']
        assert mreg['protected_aliases']['sha256']==reg['protected_aliases']['sha256']
        mc=closed_collection(Path(mreg['output_path']),mreg)
        for p in [regpath,poolpath,protected,mregpath]: before[str(p)]=sha(p)
        reports.append(dict(seed=seed,pool_sha256=sha(poolpath),pool=summary(pool['rows']),
            fixed_scheduled128=summary(chosen),chosen128_order_sha256=order,
            usable_after_protection=len(usable),ownQ=own,MC=mc))
    bookpath=Path(q['book_path']);assert sha(bookpath)==q['book_sha256']
    book=json.loads(bookpath.read_bytes())['splits']['arena']
    known=[dict(root_fen=chess.STARTING_FEN,prefix_uci=r['opening']['moves'],root_id=str(i))
           for i,r in enumerate(book)]
    assert all(sha(p)==s for p,s in before.items())
    result=dict(schema='independent-ownQ-MC-root-history-coverage-v1',status='PASS-metadata-only',
        first=first,finished=time.time(),input_sha256=before,known8_book_sha256=sha(bookpath),
        known8=summary(known),seeds=reports,actual_model_reads=0,search_calls=0,engine_calls=0,
        optimizer_updates=0,root_selection_changes=0,
        interpretation_limits=['ply<=20 and pieces<=10 are descriptive proxies, not ECO/phase labels',
            'source pool is teacher-selected ancestry; no teacher labels or models read',
            'fixed scheduled128 differs from starts actually consumed before1024 budget',
            'MC live rows/outcomes never read without closed PASS receipt',
            'coverage metadata does not measure generalization or establish causal failure',
            'piece material accounting is not an evaluator score'])
    (OUT/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'result_sha256':sha(OUT/'result.json'),'elapsed':time.time()-first,
        'summary':[{ 'seed':r['seed'],'pool_opening_le20':r['pool']['opening_ply_le20'],
         'chosen128_opening_le20':r['fixed_scheduled128']['opening_ply_le20'],
         'chosen128_low_material':r['fixed_scheduled128']['low_material_pieces_le10'],
         'ownQ_statuses':r['ownQ']['episode_statuses'],'MC':r['MC']['status']}
          for r in reports], 'known8_ply':result['known8']['histograms']['ply']}))


if __name__=='__main__':main()
