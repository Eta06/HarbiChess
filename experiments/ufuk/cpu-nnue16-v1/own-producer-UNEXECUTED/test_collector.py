import importlib.util
from pathlib import Path
from types import SimpleNamespace as N
import chess, pytest
p=Path(__file__).with_name('collector.py'); s=importlib.util.spec_from_file_location('ownq',p); c=importlib.util.module_from_spec(s); s.loader.exec_module(c)
def pool(ps): return {'schema':c.POOL,'selection_status':'pass','train_only':True,'rows':[{'root_id':f'r{i}','role':'TRAIN','root_fen':chess.STARTING_FEN,'prefix_uci':x} for i,x in enumerate(ps)]}
class F:
 evaluator=staticmethod(lambda b:.25)
 @staticmethod
 def make(fn):
  class S:
   def search(self,b):
    ms=sorted(b.legal_moves,key=lambda m:m.uci())
    for m in ms[:2]:
     b.push(m)
     try:
      if b.outcome(claim_draw=True) is None: fn(b)
     finally:b.pop()
    return N(move=ms[0],value=.25,nodes=len(ms)+3,evaluations=2,completed_depth=1,root_actions=len(ms))
  return S()
def test_mirror_and_fixed_selection():
 b=chess.Board(); b.push_uci('e2e4'); assert c.alias(b)==c.alias(b.mirror())
 x=pool([[],['e2e4'],['d2d4']]); a,h=c.starts(x,7,set(),2,3); z,j=c.starts(x,7,set(),2,3); assert [r['root_id'] for r in a]==[r['root_id'] for r in z] and h==j
def test_unknown_q_rows_no_policy_targets():
 o=c.collect(pool([[],['e2e4'],['d2d4']]),seed=7,protected=set(),factory=F(),row_limit=3,start_limit=1,plies=3,pool_size=3)
 assert len(o['training_row_ids'])==3 and o['games'][0]['status']=='unknown-row-budget-prefix'
 assert all(r['own_wdl_mover'] is None and not r['behavior_policy_available'] for r in o['rows'])
def test_terminal_wdl_mover_pov():
 class FF(F):
  @staticmethod
  def make(_):
   class S:
    def search(self,b):
     m=chess.Move.from_uci('g2g4' if b.turn else 'd8h4'); assert m in b.legal_moves
     return N(move=m,value=.25,nodes=b.legal_moves.count()+1,evaluations=0,completed_depth=1,root_actions=b.legal_moves.count())
   return S()
 o=c.collect(pool([['f2f3','e7e5']]),seed=2,protected=set(),factory=FF(),row_limit=2,start_limit=1,plies=16,pool_size=1)
 assert [r['own_wdl_mover'] for r in o['rows']]==[-1,1]
def test_protected_game_excluded():
 x=pool([[],['e2e4']]); a,_=c.starts(x,3,set(),2,2); b=c.replay(a[0]); b.push(sorted(b.legal_moves,key=lambda m:m.uci())[1])
 o=c.collect(x,seed=3,protected={c.alias(b)},factory=F(),row_limit=2,start_limit=2,plies=4,pool_size=2)
 assert o['games'][0]['status']=='excluded-protected-trajectory' and all(r['root_id']==a[1]['root_id'] for r in o['rows'] if r['train_eligible'])
def test_invalid_pool_and_sidecar(tmp_path):
 with pytest.raises(ValueError): c.starts(pool([[]]),1,set())
 with pytest.raises(ValueError,match='illegal root history'): c.starts(pool([['e2e5']]),1,set(),1,1)
 w=c.AliasWriter(tmp_path); ref=w.append([8,-3,8,1]); w.close(); raw=(tmp_path/ref['file']).read_bytes()
 assert len(raw)==24 and ref['count']==3 and __import__('struct').unpack('<3q',raw)==(-3,1,8)
