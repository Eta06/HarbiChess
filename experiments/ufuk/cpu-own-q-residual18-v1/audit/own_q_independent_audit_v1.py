import json,gzip,hashlib,math,collections,time,sys,importlib.util
from pathlib import Path
import chess
b=Path('/workspace/work/harbichess/continuation-20261005');reg=json.loads(Path(sys.argv[1]).read_text());deadline=reg['deadline_epoch'];study=Path('/workspace/HarbiChess/experiments/ufuk/cpu-classical-own-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
s=load(study/'arena/search.py','ownq_root_indep_search');v=load(study/'value.py','ownq_root_indep_value');rows=[];replays=[]
def guard():
 assert time.time()<deadline
for seed in [20262905,20262906]:
 rp=b/'own-q-reanalysis-v1-registration'/f'{seed}-registration.json';registered=json.loads(rp.read_text());path=Path(f'/dev/shm/harbichess-own-q-reanalysis/{seed}/labels-00001024.json.gz');labels=json.loads(gzip.decompress(path.read_bytes()));assert sha(path)==reg['input_pins'][str(path)];assert labels['registration_sha256']==sha(rp) and labels['seed']==seed and len(labels['roots'])==1024;assert labels['closure']==registered['closure'] and labels['search']==registered['search'];assert labels['frozen_prior_model']==v.model_dict();boards=[];identities=set();counts=collections.Counter();plies=0
 for ordinal,row in enumerate(labels['roots']):
  guard();assert row['ordinal']==ordinal and row['row_id'] not in identities;identities.add(row['row_id']);h=row['history'];board=chess.Board(h['root_fen']);assert board.is_valid();assert hashlib.sha256((h['root_fen']+'\n'+' '.join(h['prefix_uci'])).encode()).hexdigest()==row['history_sha256']
  for text in h['prefix_uci']:
   move=chess.Move.from_uci(text);assert board.is_legal(move);board.push(move);plies+=1
  assert not board.is_game_over(claim_draw=True);assert ' '.join(board.fen().split()[:4])==row['root_fen4'];assert ('white' if board.turn else 'black')==row['root_mover'];assert row['root_actions']==board.legal_moves.count();q=row['raw_q_mover'];assert math.isfinite(q) and -2<=q<=2;assert row['target_mover']==max(-1,min(1,q));assert row['exact_mate_score_range']==(abs(q)>1);assert row['zero_value_is_not_a_draw_certificate']==(q==0);assert 1<=row['nodes']<=8192 and 0<=row['evaluations']<=row['nodes'];assert ' '.join(board.fen().split()[:4]) not in registered['protected_position_keys'];counts[row['completed_depth']]+=1;boards.append(board)
 for ordinal in [0,512,1023]:
  row=labels['roots'][ordinal];board=boards[ordinal];before=(board.fen(),tuple(board.move_stack));result=s.BudgetSearch(v.ClassicalValue(),nodes=8192,quiescence_plies=2,max_depth=8,guard=guard).search(board);assert before==(board.fen(),tuple(board.move_stack));assert result.value.hex()==float(row['raw_q_mover']).hex();assert (result.nodes,result.evaluations,result.completed_depth,result.root_actions)==(row['nodes'],row['evaluations'],row['completed_depth'],row['root_actions']);assert board.is_legal(result.move);replays.append(dict(seed=seed,ordinal=ordinal,row_id=row['row_id'],move=result.move.uci(),value=result.value,nodes=result.nodes,evaluations=result.evaluations,completed_depth=result.completed_depth))
 rows.append(dict(seed=seed,roots=1024,unique_histories=len({r['history_sha256'] for r in labels['roots']}),unique_states=len({r['root_fen4'] for r in labels['roots']}),unique_trajectories=len({r['trajectory_id'] for r in labels['roots']}),legal_history_plies=plies,depth_histogram=dict(counts),total_producer_nodes=sum(r['nodes'] for r in labels['roots']),labels_sha256=sha(path)))
guard();result=dict(status='PASS-2048-full-legal-history-and-six-actual8192-search-reproductions-not-strength',rows=rows,six_actual_searches=replays,first_epoch=reg['first_epoch'],deadline_epoch=deadline,finished_epoch=time.time(),registration_sha256=sha(sys.argv[1]),auditor_sha256=sha(__file__),strength_success_claimed=False);Path(reg['result_path']).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],finished_epoch=result['finished_epoch'])))
