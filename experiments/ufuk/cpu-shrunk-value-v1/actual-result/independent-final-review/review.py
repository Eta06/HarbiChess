"""Read-only QSEARCH160 rule/identity audit and six existing-move reproductions."""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import chess

STUDY=Path('/workspace/HarbiChess/experiments/ufuk/cpu-shrunk-value-v1/arena')
ARENA=Path('/workspace/work/harbichess/cpu-shrunk-value-v1-actual/arena-recovery-v2')
OUT=Path('/workspace/work/harbichess/shrink-independent-final-review')
CHECKOUT=Path('/workspace/work/harbichess/cpu-fix-source-4cae085')


AUDIT_FIRST=time.time()
AUDIT_END=time.monotonic()+600
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})

def ORIGINAL_RESOURCE_GUARD():
    import shutil
    if time.monotonic()>=AUDIT_END:raise TimeoutError('new independent audit original600 deadline exhausted')
    if shutil.disk_usage('/workspace').free<256*1024**2:raise RuntimeError('workspace256MiB floor')

def recovered_checks(x,owner,contract,protected):
    recovery=x['recovery']
    assert recovery['old_parent_executor_sha256']==contract['inputs']['tournament.py']
    invocation=Path(recovery['parent_invocation_path']);progress=Path(recovery['parent_progress_path'])
    assert sha(invocation)==recovery['parent_invocation_sha256'] and sha(progress)==recovery['parent_progress_sha256']
    protected[str(invocation)]=sha(invocation);protected[str(progress)]=sha(progress)
    inv=json.loads(invocation.read_text())
    assert x['started_epoch']==inv['original_first_epoch'] and x['original_deadline_epoch']==inv['original_deadline_epoch']==owner['original_deadline_epoch']
    ends=[];active=None
    for line in progress.read_text().splitlines():
        e=json.loads(line)
        if e['type']=='game_start':active={'key':(e['pair'],e['color']),'moves':[]}
        elif e['type']=='move':active['moves'].append((e['ply'],e['uci']))
        elif e['type']=='game_end':ends.append(e['game']);active=None
    assert x['games'][:len(ends)]==ends and recovery['original_closed_games_reused']==len(ends)==2
    game=x['games'][len(ends)];assert (game['opening_pair'],game['candidate_color'])==active['key']
    packets=[r for r in game['search_by_move'] if r.get('measurement_origin')=='actual-recovery-prefix-research']
    assert [(r['ply'],r['selected_move']) for r in packets]==active['moves'] and len(packets)==10
    assert recovery['duplicate_NN_moves_researched']==10
    assert recovery['duplicate_NN_nodes']==sum(r['nodes'] for r in packets)==5120
    assert recovery['duplicate_NN_evaluations']==sum(r['evaluations'] for r in packets)==4745
    assert recovery['old_partial_latency_unknown'] and not recovery['SF_partial_resume_claimed']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    protocol=STUDY/'protocol.json';p=json.loads(protocol.read_text());cohort_path=ARENA/'cohort-result.json';cohort=json.loads(cohort_path.read_text());contract=cohort['contract']
    assert cohort['status']=='completed-games-not-strength'
    assert contract['inputs']['protocol.json']==sha(protocol)
    assert contract['source_commit']==p['source_commit']
    for helper in ('search.py','value.py','tournament.py'):
        assert sha(STUDY/helper)==contract['inputs'][helper]
    expected={(seed,role,opp) for seed in p['match_seeds'] for role,opp in p['tasks']}
    assert len(cohort['rows'])==10==len(expected)
    assert {(row['seed'],row['arm'],row['opponent']) for row in cohort['rows']}==expected
    book_path=Path(p['book_path']);assert sha(book_path)==p['book_sha256']
    roots=json.loads(book_path.read_text())['splits']['arena'][:8]
    protected={str(protocol):sha(protocol),str(cohort_path):sha(cohort_path),str(book_path):sha(book_path)}
    groups=[];loaded={};total_plies=0
    for owner in cohort['rows']:
        ORIGINAL_RESOURCE_GUARD()
        key=(owner['seed'],owner['arm'],owner['opponent']);seed,role,opp=key
        assert owner['status']=='completed-games-awaiting-independent-audit'
        assert owner['finished_epoch']<=owner['original_deadline_epoch']
        path=Path(owner['result_path']);assert sha(path)==owner['result_sha256'];protected[str(path)]=sha(path)
        x=json.loads(path.read_text());loaded[key]=x
        assert x['source_commit']==p['source_commit'] and x['seed']==seed and x['protocol_sha256']==sha(protocol)
        assert x['schema'] in ('cpu-all-root-quiescent-alpha-beta-arena-v1','cpu-all-root-quiescent-alpha-beta-arena-recovery-v2') and not x['GPU_used'] and not x['promotion_ready']
        assert x['search_nodes_per_move']==512 and x['quiescence_plies']==2 and x['max_plies']==400
        assert x['opening_source_sha256']==p['book_sha256'] and x['finished_epoch']<=x['original_deadline_epoch']==owner['original_deadline_epoch']
        for header,helper in [('helper_sha256','tournament.py'),('search_helper_sha256','search.py'),('value_helper_sha256','value.py')]:
            expected_helper=contract['recovery_executor_sha256'] if x['schema'].endswith('recovery-v2') and helper=='tournament.py' else contract['inputs'][helper]
            assert x[header]==expected_helper
        if x['schema'].endswith('recovery-v2'):
            recovered_checks(x,owner,contract,protected)
        for model_role,field in [(role,'candidate_sha256')]+([] if opp=='SF512' else [(opp,'opponent_sha256')]):
            model=p['models'][str(seed)][model_role];model_path=Path(model['path']);assert sha(model_path)==model['sha256']==x[field];protected[str(model_path)]=sha(model_path)
        if opp=='SF512':assert (x['stockfish_sha256'],x['stockfish_nodes'],x['stockfish_threads'],x['stockfish_hash_mib'])==(p['stockfish_sha256'],512,1,16)
        assert len(x['games'])==16
        seen=set();scores=[];sf_nodes=[];nn_nodes=[];nn_evaluations=[];caps=0;group_plies=0
        for game in x['games']:
            game_key=(game['opening_pair'],game['candidate_color']);assert game_key not in seen;seen.add(game_key)
            pair,color=game_key;assert 0<=pair<8 and color in ('white','black');white=color=='white'
            opening=roots[pair]['opening']['moves'];assert game['opening']==opening==game['moves'][:len(opening)]
            nn={r['ply']:r for r in game['search_by_move']};sf={r['ply']:r for r in game['stockfish_nodes_by_move']}
            assert len(nn)==len(game['search_by_move']) and len(sf)==len(game['stockfish_nodes_by_move'])
            board=chess.Board();expected_nn=[];expected_sf=[]
            for index,uci in enumerate(game['moves']):
                move=chess.Move.from_uci(uci);assert board.is_legal(move)
                if index>=len(opening):
                    assert board.outcome(claim_draw=True) is None
                    ply=index+1
                    if board.turn==white or opp!='SF512':
                        expected_nn.append(ply);r=nn[ply]
                        assert r['candidate']==(board.turn==white) and r['selected_move']==uci
                        assert type(r['nodes']) is int and type(r['root_actions']) is int
                        assert r['root_actions']==r['legal_root_actions']==board.legal_moves.count()
                        assert r['root_actions']+1<=r['nodes']<=512 and 0<=r['evaluations']<=r['nodes'] and 1<=r['completed_depth']<=8
                        assert math.isfinite(r['value']) and -2<=r['value']<=2
                        assert math.isfinite(r['wall_seconds']) and r['wall_seconds']>=0
                        nn_nodes.append(r['nodes']);nn_evaluations.append(r['evaluations'])
                    else:
                        expected_sf.append(ply);r=sf[ply];assert type(r['nodes']) is int and r['nodes']>=0
                        assert math.isfinite(r['wall_seconds']) and r['wall_seconds']>=0;sf_nodes.append(r['nodes'])
                board.push(move)
            assert sorted(nn)==expected_nn and sorted(sf)==expected_sf
            assert board.ply()==game['plies']
            assert len(game['move_wall_seconds'])==len(game['moves'])-len(opening)
            assert all(math.isfinite(t) and t>=0 for t in game['move_wall_seconds'])
            outcome=board.outcome(claim_draw=True)
            if outcome is None:
                assert game['plies']==400 and game['termination']=='max_plies';caps+=1;score=.5
            else:
                assert game['termination']==outcome.termination.name.lower();score=.5 if outcome.winner is None else float(outcome.winner==white)
            assert game['score']==score;scores.append(score);group_plies+=len(game['moves'])-len(opening)
        assert seen=={(pair,color) for pair in range(8) for color in ('white','black')}
        wins=sum(s==1 for s in scores);draws=sum(s==.5 for s in scores);losses=16-wins-draws;mean=sum(scores)/16
        assert (x['summary']['wins'],x['summary']['draws'],x['summary']['losses'],x['summary']['score'],x['summary']['capped_games'])==(wins,draws,losses,mean,caps)
        total_plies+=group_plies
        groups.append(dict(seed=seed,role=role,opponent=opp,games=16,wins=wins,draws=draws,losses=losses,score=mean,caps=caps,continuation_plies=group_plies,NN_moves=len(nn_nodes),NN_nodes=sum(nn_nodes),NN_max=max(nn_nodes,default=0),NN_evaluations=sum(nn_evaluations),SF_moves=len(sf_nodes),SF_actual_nodes=sum(sf_nodes),SF_over512_moves=sum(n>512 for n in sf_nodes),SF_actual_max=max(sf_nodes,default=0)))
    assert all(sha(Path(path))==digest for path,digest in protected.items())
    return p,loaded,dict(schema='shrink-independent-full160-rule-audit-v2',status='PASS-full160-integrity-NOT-strength-qualification',games=160,groups=groups,continuation_plies=total_plies,original_first_epoch=contract['original_first_epoch'],original_deadline_epoch=contract['original_deadline_epoch'],actual_SF_over512_moves=sum(r['SF_over512_moves'] for r in groups),actual_SF_max=max(r['SF_actual_max'] for r in groups),protected_inputs_results_models_sha256=protected,source_commit=p['source_commit'],protocol_sha256=sha(protocol),actual_at_epoch=time.time(),GPU_used=False,new_training_updates=0,new_games=0,limitations=['NNrecursivecounts vsStockfish actualnodecounts are NOT equalcompute.','Rootcoverage uses protectedcode + legalrootcounters; sixNNmovepackets independentlyreproduced, noteveryNNpacket.','Development8knownfamilies, no virginformalstrength/CI/latency qualification.'])


def reproduce(p,loaded):
    sys.path[:0]=[str(STUDY),str(CHECKOUT/'src')]
    import torch
    from search import BudgetSearch
    from value import NeuralValue
    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    deadline=AUDIT_END
    def guard():
        if time.monotonic()>=deadline:raise TimeoutError('boundedsixmovepacket audit exhausted')
    reports=[]
    for role,opponent in [('e8','SF512'),('rebased','SF512'),('shrunk','e8')]:
        x=loaded[20262605,role,opponent];model=Path(p['models']['20262605'][role]['path']);before=sha(model)
        evaluator=NeuralValue(model);search=BudgetSearch(evaluator,nodes=512,quiescence_plies=2,max_depth=8,guard=guard)
        selected=[]
        for game in x['games']:
            selected += [(game,r) for r in game['search_by_move'] if r['candidate']]
            if len(selected)>=2:break
        assert len(selected)>=2 and all(g['opening_pair']==0 and g['candidate_color']=='white' for g,r in selected[:2])
        for game,packet in selected[:2]:
            guard();board=chess.Board()
            for move in game['moves'][:packet['ply']-1]:board.push_uci(move)
            ORIGINAL_RESOURCE_GUARD()
            original=board.fen(),tuple(board.move_stack);start=time.perf_counter();actual=search.search(board)
            observed={'selected_move':actual.move.uci(),'value':actual.value,'nodes':actual.nodes,'evaluations':actual.evaluations,'completed_depth':actual.completed_depth,'root_actions':actual.root_actions}
            expected={key:packet[key] for key in observed}
            assert observed==expected and (board.fen(),tuple(board.move_stack))==original
            reports.append(dict(role=role,seed=20262605,opening_pair=game['opening_pair'],candidate_color=game['candidate_color'],ply=packet['ply'],fullhistory_plies=board.ply(),mover='white' if board.turn else 'black',recorded_equals_reproduced=True,actual_packet=observed,model_sha256=before,wall_seconds=time.perf_counter()-start))
        assert sha(model)==before
    return dict(schema='shrink-six-fixed-existing-neural-packet-reproduction-v1',status='PASS-six-exact-packets-NOT-new-games',source_commit=p['source_commit'],torch_version=torch.__version__,threads=1,rows=reports,matched_packets=6,actual_at_epoch=time.time(),read_only_models=True,new_games=0,new_training_updates=0,GPU_used=False,helper_sha256={name:sha(STUDY/name) for name in ('search.py','value.py')})


if __name__=='__main__':
    p,loaded,receipt=audit();packets=reproduce(p,loaded)
    screens=[]
    for seed in p['match_seeds']:
        score=lambda role,opp:next(r['score'] for r in receipt['groups'] if (r['seed'],r['role'],r['opponent'])==(seed,role,opp))
        direct=score('shrunk','e8');learning=score('shrunk','rebased');sf=score('shrunk','SF512');base=score('e8','SF512');zero=score('rebased','SF512')
        gates={'direct_same_search_e8_gt_060':direct>.6,'pairedSF_gain_over_e8_gt_010':sf-base>.1,'finalSF_ge_025':sf>=.25,'trained_vs_untrained_gt_060':learning>.6,'pairedSF_gain_over_untrained_gt_0':sf-zero>0,'caps_le_005':all(r['caps']/16<=.05 for r in receipt['groups'] if r['seed']==seed)}
        screens.append({'seed':seed,'direct_e8_score':direct,'trained_vs_zero_score':learning,'finalSF_score':sf,'e8SF_score':base,'zeroSF_score':zero,'gates':gates,'passed_screen':all(gates.values())})
    receipt['screens']=screens;receipt['screen_result']='PASS-development-only' if all(s['passed_screen'] for s in screens) else 'FAIL'
    receipt['original_recovery_prefix_proof_paths']=[str(path) for path in Path('/workspace/work/harbichess/shrink-recovery-review').glob('*.json')]
    receipt['recovery_prefix_scope']='previous10move research proof and current6move model comparisons are distinct; no claim all NN searches independently recomputed'
    receipt['audit_original_first_epoch']=AUDIT_FIRST;receipt['audit_original_deadline_epoch']=AUDIT_FIRST+600
    receipt['actual_SF_nodes']=sum(r['SF_actual_nodes'] for r in receipt['groups']);receipt['NN_nodes']=sum(r['NN_nodes'] for r in receipt['groups'])

    for name,value in [('full160-rule-audit.json',receipt),('six-existing-packets.json',packets)]:
        path=OUT/name
        with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
    assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<128*1024
    print(json.dumps({'games':receipt['games'],'continuation_plies':receipt['continuation_plies'],'neural_packets_exact':packets['matched_packets'],'SF_over512':receipt['actual_SF_over512_moves'],'SF_max':receipt['actual_SF_max']}))
