"""Read-only full8192 legal/dataset alignment witness. No NN forward or SGD."""
import os
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import gc
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

ROOT = Path('/workspace/HarbiChess')
OUT = Path(__file__).resolve().parent
MAIN = ROOT / 'experiments/ufuk'
CORE = Path('/workspace/work/harbichess/cpu-additive-source-6fcc8b4')
sys.path.insert(0, str(CORE/'src'))
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
START = time.time()
DEADLINE = min(START+900, 1791273600)


def guard():
    if time.time() >= DEADLINE:
        raise TimeoutError('read-only900deadline')


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(p):
    return digest(p.read_bytes())


def load(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


feature_path = MAIN/'cpu-fresh-learning-v1/features.py'
features = load(feature_path, 'features')
search = load(MAIN/'cpu-fresh-sc-v1/own_search_consistency.py', 'own_search_consistency')
mc = load(MAIN/'cpu-fresh-learning-v1/train.py', 'main_mc')
sc = load(MAIN/'cpu-fresh-sc-v1/train.py', 'main_sc')
journal_path = MAIN/'cpu-fresh-selfplay-v2/journal_v2.py'
journal = load(journal_path, 'journal_actual')
full = load(MAIN/'cpu-fresh-fullcritic-v1/core.py', 'full_actual_core')
from harbichess.training.torch_array_encoder import TorchArrayBoardEncoder


def independently_replay(state, config):
    rng = random.Random(config['seed'])
    x, y, anchors, groups_by_game, unique = [], [], [], {}, {}
    protected = set(config['excluded_training_position_keys'])
    exclusion_ledger = []
    duplicate_games = duplicate_rows = protected_games = protected_rows = 0
    known_games = unknown_games = unknown_rows = active_rows = total = 0
    wdls = {}
    games = state['games'] + ([state['active']] if state['active'] is not None else [])
    for gi, game in enumerate(games):
        guard()
        ri = rng.randrange(len(config['roots']))
        assert game['root_index'] == ri and game['game_index'] == gi
        root = config['roots'][ri]
        board = chess.Board(root['root_fen'])
        assert board.is_valid()
        touched = {' '.join(board.fen().split()[:4])}
        for text in root['prefix']:
            move = chess.Move.from_uci(text)
            assert move in board.legal_moves
            board.push(move)
            touched.add(' '.join(board.fen().split()[:4]))
        positions = []
        for row in game['moves']:
            assert board.outcome(claim_draw=True) is None and board.ply() < 400
            legal = sorted(board.legal_moves, key=lambda m: m.uci())
            selected = chess.Move.from_uci(row['selected'])
            assert selected in legal
            explored = rng.random() < .05
            played = legal[rng.randrange(len(legal))] if explored else selected
            mover = 'white' if board.turn else 'black'
            assert (row['action'], row['explored'], row['mover'], row['pre_ply'],
                    row['legal_count'], row['root_actions'], row['mu']) == (
                        played.uci(), explored, mover, board.ply(), len(legal), len(legal),
                        .05/len(legal)+(.95 if played == selected else 0))
            anchor = np.asarray(row['e8_anchor_wdl'], dtype=np.float32)
            assert anchor.shape == (3,) and np.isfinite(anchor).all() and (anchor >= 0).all()
            assert abs(float(anchor.sum())-1) < 2e-6
            positions.append((features.invariants(board), mover, anchor))
            board.push(played)
            touched.add(' '.join(board.fen().split()[:4]))
            total += 1
        outcome = board.outcome(claim_draw=True)
        is_closed = gi < len(state['games'])
        if is_closed:
            expected = board.result(claim_draw=True) if outcome else 'UNKNOWN'
            termination = outcome.termination.name if outcome else 'total-ply-cap'
            assert game['result'] == expected and game['termination'] == termination
            assert outcome is not None or board.ply() >= 400
        else:
            assert outcome is None and board.ply() < 400
            active_rows += len(positions)
            expected = 'ACTIVE'
        trajectory = digest(canonical({'root_fen': root['root_fen'], 'prefix': root['prefix'],
                                       'moves': [row['action'] for row in game['moves']]}))
        known = is_closed and expected != 'UNKNOWN'
        if known:
            known_games += 1
            start = len(y)
            for row_x, mover, anchor in positions:
                won = (expected == '1-0' and mover == 'white') or (
                    expected == '0-1' and mover == 'black')
                label = 1 if expected == '1/2-1/2' else 0 if won else 2
                x.append(row_x); y.append(label); anchors.append(anchor)
                key = f'{expected}/{mover}/{label}'
                wdls[key] = wdls.get(key, 0)+1
            groups_by_game[gi] = tuple(range(start, len(y)))
        elif is_closed:
            unknown_games += 1
            unknown_rows += len(positions)
        if not is_closed:
            continue  # Active never eligible, even if future eventual outcome is known.
        if touched & protected:
            protected_games += 1
            n = len(groups_by_game.get(gi, ()))
            protected_rows += n
            exclusion_ledger.append(dict(game_index=gi, trajectory_sha256=trajectory,
                                         known_rows_removed=n, result_status=game['result'],
                                         termination=game['termination']))
        elif known:
            if trajectory in unique:
                duplicate_games += 1
                duplicate_rows += len(groups_by_game[gi])
            else:
                unique[trajectory] = groups_by_game[gi]
    assert total == state['actions'] == 8192
    assert canonical(rng.getstate()) == canonical(state['rng'])
    keys = tuple(sorted(unique))
    groups = tuple(unique[k] for k in keys)
    train_g = tuple(i for i, k in enumerate(keys) if int(k[-2:],16)%5 != 0)
    val_g = tuple(i for i, k in enumerate(keys) if int(k[-2:],16)%5 == 0)
    train = tuple(i for gi in train_g for i in groups[gi])
    val = tuple(i for gi in val_g for i in groups[gi])
    assert train and val and not set(train)&set(val)
    return (np.asarray(x,dtype=np.float32), np.asarray(y,dtype=np.int64),
            np.asarray(anchors,dtype=np.float32), groups, keys, train_g, val_g, train, val), {
                'raw_known_rows':len(y),'known_completed_games':known_games,
                'unknown_capped_games':unknown_games,'unknown_cap_rows':unknown_rows,
                'unfinished_rows':active_rows,'protected_complete_games':protected_games,
                'protected_known_rows':protected_rows,'duplicate_trajectories':duplicate_games,
                'duplicate_known_rows':duplicate_rows,'protected_ledger_sha256':digest(canonical(exclusion_ledger)),
                'mover_label_counts':wdls}


result = dict(schema='fresh-final8192-independent-readiness-v1', status='failed-preserved',
              started_epoch=START, deadline_epoch=DEADLINE, cpu_threads=1,
              NN_forward_calls=0, optimizer_steps=0, rows=[])
try:
    for seed in (20262805,20262806):
        p=Path(f'/dev/shm/harbichess-fresh-E0-{seed}/actions-00008192.json.gz')
        cfg=Path(f'/workspace/work/harbichess/cpu-fresh-selfplay-v2-actual/registration/{seed}-E0-actor-config.json')
        js, cs = sha(p), sha(cfg)
        config=json.loads(cfg.read_bytes()); state=journal.read(p)
        assert state['phase']=='epoch-action-ceiling' and config['max_actions']==8192
        assert config['original_deadline_epoch']==1791235580.0397933
        actor_receipt=Path(f'/workspace/work/harbichess/cpu-fresh-selfplay-v2-actual/{seed}-E0/actions-00008192.process-result.json')
        receipt=json.loads(actor_receipt.read_bytes())
        # Record receipt keys without assuming the producer's wrapper field names.
        independent, counts=independently_replay(state,config)
        common=mc.prepare_fresh(p,js,cfg,cs,journal_path,sha(journal_path),
                                feature_path,sha(feature_path),
                                protected_position_keys=config['excluded_training_position_keys'])
        sc_common=sc.prepare_fresh(p,js,cfg,cs,journal_path,sha(journal_path),
                                   feature_path,sha(feature_path),
                                   protected_position_keys=config['excluded_training_position_keys'])
        for i in range(3):
            bits(common[i].numpy(),independent[i]);bits(common[i].numpy(),sc_common[i].numpy())
        for i in range(3,9):
            assert common[i]==independent[i]==sc_common[i]
        assert common[9:]==sc_common[9:]
        split_receipt=common[9][1]
        assert split_receipt['protected_exclusion_ledger_sha256']==counts['protected_ledger_sha256']
        assert split_receipt['deduplicated_identical_trajectory_aliases']==counts['duplicate_trajectories']
        assert split_receipt['protected_known_rows_removed']==counts['protected_known_rows']
        expected_data_sha=digest(common[0].numpy().tobytes()+common[1].numpy().tobytes()
                                 +common[2].numpy().tobytes()+canonical(independent[3])
                                 +canonical(independent[4])+canonical(common[9]))
        assert common[10]==expected_data_sha
        extracted=search.extract_verified(p,js,config,journal,feature_path,sha(feature_path),guard)
        attached=search.bind_common_data(extracted,common)
        dense=full.extract_full_history_rows(state,config,journal,TorchArrayBoardEncoder(),
                                             features.invariants,len(common[1]))
        full_binding=full.bind_common_rows(common,dense,attached)
        assert full_binding['common_dataset_sha256']==common[10]
        train_rows,len_val=len(common[7]),len(common[8])
        split_sha=digest(canonical({'train':common[7],'validation':common[8]}))
        info=dict(seed=seed,actions=8192,journal_sha256=js,actor_config_sha256=cs,
                  actor_source_commit=config['source_commit'],actor_receipt_sha256=sha(actor_receipt),
                  actor_process_receipt=receipt,counts=counts,
                  train_rows=train_rows,validation_rows=len_val,
                  eligible_rows=sum(map(len,common[3])),eligible_trajectories=len(common[3]),
                  train_trajectories=len(common[5]),validation_trajectories=len(common[6]),
                  updates=min(1024,4*train_rows//256),common_MC_dataset_sha256=common[10],
                  split_sha256=split_sha,SC_extended_dataset_sha256=attached['search_extended_data_sha256'],
                  FULL_dense_search_rows_sha256=full_binding['search_rows_sha256'],
                  labels_sha256=digest(common[1].numpy().tobytes()),anchors_sha256=digest(common[2].numpy().tobytes()),
                  x840_sha256=digest(common[0].numpy().tobytes()),x104_sha256=digest(dense['x104'].tobytes()),
                  SC_normalized_search_sha256=digest(extracted['normalized_search'].tobytes()),
                  explicit_scope='Independent legal/mover/outcome/RNG/fullhistory/protected/dedup/split reconstruction; MAIN MC/SC and FULLdense commondata alignment. Stored anchors byte-exact, E8 identity pinned; NO neural anchor/Q recomputation or strength claim.')
        assert sha(p)==js and sha(cfg)==cs
        result['rows'].append(info)
        print(json.dumps({'seed':seed,'status':'PASS','train':train_rows,'val':len_val,
                          'updates':info['updates']}),flush=True)
        del common,sc_common,independent,extracted,attached,dense,state
        gc.collect()
    result['status']='PASS-final8192-readonly-data-readiness-not-fit-or-strength'
except Exception as e:
    result['error']=repr(e)
    raise
finally:
    result['finished_epoch']=time.time()
    result['elapsed_seconds']=time.time()-START
    result['auditor_sha256']=sha(Path(__file__))
    result['source_sha256']={str(p):sha(p) for p in [MAIN/'cpu-fresh-learning-v1/train.py',
      MAIN/'cpu-fresh-sc-v1/train.py',MAIN/'cpu-fresh-sc-v1/own_search_consistency.py',
      MAIN/'cpu-fresh-fullcritic-v1/core.py',feature_path,journal_path,
      CORE/'src/harbichess/training/torch_array_encoder.py']}
    (OUT/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
