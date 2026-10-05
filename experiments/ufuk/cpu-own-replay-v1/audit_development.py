"""Fresh-process full-history audit of CPU pilot development games; no engine queries."""
import hashlib,json,statistics
from pathlib import Path
import chess
R=Path('/workspace/work/harbichess/cpu-own-replay-v1-actual/development');M=Path('/workspace/HarbiChess');protocol=json.loads((M/'experiments/ufuk/cpu-own-replay-v1/development-protocol.json').read_text());book=json.loads((R/'book.json').read_text());rows=[];total_plies=0;total_nodes=0
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for seed in protocol['seeds']:
 for arm,opponent in [('value-only','e8'),('e8','SF512'),('value-only','SF512')]:
  p=R/f'{seed}-{arm}-vs-{opponent}.json';x=json.loads(p.read_text());assert x['seed']==seed and x['source_commit']==protocol['source_commit'];assert x['neural_simulations_per_move']==16 and x['neural_threads']==1 and x['candidate_root_actions']==4 and x['max_plies']==400;assert not x['promotion_ready'];assert x['opening_source_sha256']==sha(R/'book.json')
  if arm=='e8':expected=protocol['initial_e8_sha256']
  else:expected=sha(R.parent/f'{seed}-value-only/final-attempt-00000080/model.safetensors')
  assert x['candidate_sha256']==expected
  if opponent=='e8':assert x['opponent_sha256']==protocol['initial_e8_sha256'] and x['opponent_root_actions']==4
  else:assert x['stockfish_sha256']==protocol['stockfish_sha256'] and (x['stockfish_nodes'],x['stockfish_threads'],x['stockfish_hash_mib'])==(512,1,16)
  assert len(x['games'])==16
  pairs=set();ply_count=nodes=0
  for g in x['games']:
   key=(g['opening_pair'],g['candidate_color']);assert key not in pairs;pairs.add(key);assert g['candidate_color'] in ('white','black');opening=book['splits']['arena'][g['opening_pair']]['opening']['moves'];assert g['opening']==opening and g['moves'][:len(opening)]==opening
   board=chess.Board();sf_plies=[]
   for i,u in enumerate(g['moves']):
    if i>=len(opening):
     assert board.outcome(claim_draw=True) is None
     if opponent=='SF512' and board.turn != (g['candidate_color']=='white'):sf_plies.append(i+1)
    move=chess.Move.from_uci(u);assert move in board.legal_moves;board.push(move)
   assert g['plies']==len(g['moves'])
   outcome=board.outcome(claim_draw=True)
   if outcome is None:assert g['termination']=='max_plies' and g['plies']==400 and g['score']==0.5
   else:
    name=outcome.termination.name.lower();assert g['termination']==name,(name,g['termination']);score=0.5 if outcome.winner is None else float(outcome.winner==(g['candidate_color']=='white'));assert g['score']==score
   assert len(g['move_wall_seconds'])==len(g['moves'])-len(opening) and all(t>=0 for t in g['move_wall_seconds'])
   if opponent=='SF512':
    receipt=g['stockfish_nodes_by_move'];assert [n['ply'] for n in receipt]==sf_plies and all(isinstance(n['nodes'],int) and n['nodes']>=0 for n in receipt);nodes+=sum(n['nodes'] for n in receipt)
   ply_count+=len(g['moves'])-len(opening)
  assert pairs=={(i,c) for i in range(8) for c in ('white','black')};summary=x['summary'];score=sum(g['score'] for g in x['games'])/16;assert score==summary['score'];assert summary['capped_games']==sum(g['termination']=='max_plies' for g in x['games']);assert nodes==x['stockfish_actual_nodes'];total_plies+=ply_count;total_nodes+=nodes
  rows.append({'seed':seed,'arm':arm,'opponent':opponent,'path':str(p),'sha256':sha(p),'score':score,'wins':summary['wins'],'draws':summary['draws'],'losses':summary['losses'],'unknown_caps':summary['capped_games'],'bootstrap_pair_95':summary['bootstrap_pair_95'],'hoeffding_pair_95':summary['hoeffding_pair_95'],'legal_continuation_plies':ply_count,'actual_SF_nodes':nodes,'wall_seconds':x['wall_seconds'],'positions_per_wall_second':x['positions_per_wall_second']})
contrast=[]
for seed in protocol['seeds']:
 direct=next(r for r in rows if r['seed']==seed and r['opponent']=='e8');final=next(r for r in rows if r['seed']==seed and r['arm']=='value-only' and r['opponent']=='SF512');base=next(r for r in rows if r['seed']==seed and r['arm']=='e8');gates={'direct_gt_060':direct['score']>0.60,'SF_gain_gt_010':final['score']-base['score']>0.10,'finalSF_ge_025':final['score']>=0.25,'caps_le_005':all(r['unknown_caps']/16<=0.05 for r in (direct,final,base))};contrast.append({'seed':seed,'pairedSF_mean_gain':final['score']-base['score'],'screen_gates':gates,'passed_screen':all(gates.values())})
result={'schema':'cpu-own-replay-development-fullhistory-audit-v1','status':'PASS-fullhistory-integrity-screen-FAIL','rows':rows,'contrasts':contrast,'games':96,'legal_continuation_plies':total_plies,'actual_SF_nodes':total_nodes,'scope':'known development suite, not independent confirmation; empirical degenerate bootstrap does not prove a population probability0 or game equivalence','joint':'0updates and allparameters exact initialE8;64 duplicate games notrun as disclosed futilitydeviation','strength_success_claimed':False,'method_result':'FAIL both-seed development screen','source_commit':protocol['source_commit'],'protocol_sha256':sha(M/'experiments/ufuk/cpu-own-replay-v1/development-protocol.json'),'audit_helper_sha256':sha(Path(__file__)),'GPU_used':False}
(R/'independent-fullhistory-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','games','legal_continuation_plies','actual_SF_nodes','contrasts']}))
