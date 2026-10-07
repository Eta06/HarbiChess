"""Metadata-only, fail-closed NNUE own-search registration builder; never loads weights."""
import argparse, hashlib, json, struct, subprocess, time
from pathlib import Path
import chess

HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/HarbiChess/experiments/ufuk')
SEEDS=(20262905,20262906)
CORE='/workspace/work/harbichess/cpu-additive-source-6fcc8b4'
COMMIT='6fcc8b476d25495d1c9c413e55b2c7ba4794013e'
PROFILE=ROOT/'wrap-20261006/NNUE-teacher-trained-profile-v2.json'
PROFILE_SHA='daface422c690e1e989599fb8378317c9bf37f2d773ee5977e08a7f8273179d9'
PARENT=ROOT/'cpu-nnue16-v1'
EXT=Path('/workspace/work/harbichess/cpu-kingbucket-nnue-proposal/_kingbucket16.cpython-312-x86_64-linux-gnu.so')
SEARCH=ROOT/'cpu-budget-search-v1/search.py'
SEARCH_SHA='de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670'
LABELDIR=Path('/dev/shm/harbichess-NNUE-teacher8192-v1')
LINEAGE={20262905:('b73cd739fb0b357e6c57f26452ddec229f2f886936a534b467e4bc8e100024b9',
  '/workspace/work/harbichess/continuation-20261005/own-deeper-action-v1-registration/20262905-registration.json'),
 20262906:('e4bdb4cc6da1a3688cc6d1f1394cfe786c66c55e3479868f30733240fc9b30bf',
  '/workspace/work/harbichess/continuation-20261005/own-deeper-action-v1-registration/20262906-registration.json')}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canon(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def hex64(s): return isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)
def write_once(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f: f.write(data); f.flush(); __import__('os').fsync(f.fileno())
def exact_once(path,data):
    path=Path(path)
    if path.exists():
        if path.read_bytes()!=data: raise ValueError('existing immutable pool bytes differ')
    else: write_once(path,data)
def project_selection(seed, selection, labels):
    selection=Path(selection); labels=Path(labels)
    expected=LABELDIR/str(seed)
    if selection.resolve()!=expected/'selection.json' or labels.resolve()!=expected/'labels-00004096.json.gz':
        raise ValueError('exact final teacher TRAIN selection/label paths required')
    sel=json.loads(selection.read_bytes()); receipt=sel.get('receipt',{}); rows=sel.get('rows')
    if len(rows or [])!=4096 or receipt.get('selected_roots')!=4096 or not labels.is_file():
        raise ValueError('final teacher selection must contain exactly4096 roots')
    if receipt.get('original_validation_games',0)<1 or receipt.get('selected_trajectories',0)<1:
        raise ValueError('teacher TRAIN/VAL lineage receipt required')
    lineage=receipt.get('original_lineage',{}); expected_lineage=LINEAGE[seed]
    if (lineage.get('path')!=expected_lineage[1] or lineage.get('sha256')!=expected_lineage[0]
            or sha(lineage['path'])!=expected_lineage[0]):
        raise ValueError('exact inherited own-training/VAL source registration')
    projected=[]; ids=set(); histories=set()
    for r in rows:
        if set(('root_fen','prefix_uci','row_id','trajectory_id','history_sha256'))-r.keys():
            raise ValueError('root source row fields missing')
        b=chess.Board(r['root_fen']); prefix=r['prefix_uci']
        if not isinstance(prefix,list) or not prefix: raise ValueError('nonempty full history required')
        for uci in prefix:
            m=chess.Move.from_uci(uci)
            if m not in b.legal_moves: raise ValueError('illegal teacher-selected root history')
            b.push(m)
        h=hashlib.sha256((r['root_fen']+'\n'+' '.join(prefix)).encode()).hexdigest()
        if (not b.is_valid() or b.outcome(claim_draw=True) or h!=r['history_sha256']
                or ' '.join(b.fen().split()[:4])!=r['fen4']):
            raise ValueError('root history/FEN/fullhistory digest mismatch')
        if r['row_id'] in ids or h in histories: raise ValueError('duplicate selected root/history')
        ids.add(r['row_id']); histories.add(h)
        projected.append(dict(root_id=r['row_id'],source_row_id=r['row_id'],
            trajectory_id=r['trajectory_id'],role='TRAIN',root_fen=r['root_fen'],prefix_uci=prefix.copy()))
    if len({r['trajectory_id'] for r in projected})!=receipt['selected_trajectories']:
        raise ValueError('trajectory lineage count mismatch')
    pool=dict(schema='teacher-selected-ownq-train-roots-v2',selection_status='pass',train_only=True,
        source_selection_sha256=sha(selection),source_teacher_labels_sha256=sha(labels),
        source_teacher_receipt=receipt,rows=projected)
    return pool


def build(control_path):
    cp=Path(control_path).resolve(); c=json.loads(cp.read_bytes())
    required={'schema','status','seed','original_first_epoch','original_deadline_epoch','operator_end_epoch',
      'core_repo','core_commit','cpu_core','root_pool_output','protected_aliases','parent_candidate',
      'output_path','registration_output'}
    if set(c)!=required: raise ValueError('operator control has missing/extra fields')
    if c.get('schema')!='own-nnue-ownq-v2-root-control' or c.get('status')!='registered':
        raise ValueError('ROOT registered operator control required')
    seed=c.get('seed')
    if seed not in SEEDS: raise ValueError('fixed teacher seed')
    first,end,operator_end=(c.get(k) for k in ('original_first_epoch','original_deadline_epoch','operator_end_epoch'))
    if (type(first) not in (int,float) or type(end) not in (int,float) or type(operator_end) not in (int,float)
            or not first<end<=first+7200 or not end<=operator_end or not first<=time.time()<end):
        raise ValueError('fresh ROOT observed 7200s phase clock/operator end')
    if c.get('core_repo')!=CORE or c.get('core_commit')!=COMMIT or c.get('cpu_core') not in range(4):
        raise ValueError('exact clean runtime pin and CPU core')
    if c.get('output_path')!=f'/dev/shm/harbichess-ownq-v2/{seed}':
        raise ValueError('exact immutable RAM output path')
    if c.get('root_pool_output')!=f'/workspace/work/harbichess/continuation-20261007/nnue-own-producer-v2/input-pools/{seed}.json':
        raise ValueError('exact metadata pool path')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=CORE,text=True).strip()!=COMMIT or subprocess.check_output(['git','status','--porcelain'],cwd=CORE,text=True):
        raise ValueError('clean core checkout')
    if sha(PROFILE)!=PROFILE_SHA: raise ValueError('frozen teacher admission profile SHA')
    profile=json.loads(PROFILE.read_bytes())
    admission=next((x for x in profile['admissions'] if x['seed']==seed),None)
    parent=c.get('parent_candidate',{}); pp=Path(f'/dev/shm/harbichess-NNUE-teacher-fit-v1/{seed}/whole/candidate.pt')
    if set(parent)!={'path','sha256','contract_sha256'}: raise ValueError('exact parent candidate inventory')
    if (profile.get('status')!='PASS-trained-teacher-profile-not-selflearning'
        or profile.get('strength_success_claimed') is not False
        or admission is None or admission['proof_fresh_loads']!=6 or admission['fit_fresh_loads']!=2
        or admission['strict_native_loads']!=8 or admission['teacher_is_selflearning'] is not False
        or parent.get('path')!=str(pp) or parent.get('sha256')!=admission['teacher_candidate_sha256']
        or not hex64(parent.get('contract_sha256'))):
        raise ValueError('same-seed teacher parent must pass frozen proof/fit and exact candidate admission')
    selection=LABELDIR/str(seed)/'selection.json'; labels=LABELDIR/str(seed)/'labels-00004096.json.gz'
    pool=project_selection(seed,selection,labels)
    dest=Path(c['root_pool_output']).resolve()
    if not dest.is_relative_to(Path('/workspace/work/harbichess/continuation-20261007/nnue-own-producer-v2/input-pools')):
        raise ValueError('publish-once bounded metadata output path')
    exact_once(dest,canon(pool)+b'\n')
    protected=c.get('protected_aliases',{})
    if set(protected)!={'path','sha256','manifest_path','manifest_sha256'}: raise ValueError('exact protection inventory')
    binary=Path(protected.get('path','')).resolve(); manifest=Path(protected.get('manifest_path','')).resolve()
    if sha(binary)!=protected.get('sha256') or binary.stat().st_size%8 or binary.stat().st_size>256*2**20:
        raise ValueError('canonical protected int64 alias input binding')
    raw=binary.read_bytes(); vals=list(struct.unpack(f'<{len(raw)//8}q',raw))
    if vals!=sorted(set(vals)): raise ValueError('protected aliases must be sorted unique int64')
    if sha(manifest)!=protected.get('manifest_sha256'): raise ValueError('protected source manifest SHA')
    pm=json.loads(manifest.read_bytes())
    if (pm.get('schema')!='ownq-protected-aliases-manifest-v1' or pm.get('status')!='PASS-complete'
            or pm.get('alias_binary_sha256')!=sha(binary)
            or pm.get('alias_encoding')!='canonical-mirror-boardfen-sha256-64le-v1'
            or not {'known8','validation','root_manifest'}.issubset(set(pm.get('source_roles',[])))):
        raise ValueError('known8/VAL/root-manifest protected projection incomplete')
    for src in pm.get('sources',[]):
        if sha(src['path'])!=src['sha256']: raise ValueError('protected source changed')
    helper={
      'directory':str(PARENT),'model_sha256':'c04914bc5a560ad51de81c6d0aaeb3fae4d04b45c07cd8e39bbd31f655d36565',
      'native_sha256':'f60a1f9814f48691af1e2d1f1db1b548cff3e7df0a7cae9718051de1865b8b69',
      'evaluator_sha256':'5317e736f872a2e237e45942c8df9e797c5acee8f626bb12009464869fab8d95',
      'prior_path':str(ROOT/'cpu-classical-own-v1/value.py'),'prior_sha256':'a99cddfc397bd9221b99a70691427fdd663488b033fb239d9f48240786b2f277',
      'extension_path':str(EXT),'extension_sha256':'495505cdebd6dbc029ee314846dbac79d5cca6d445de26c804a43821808d644b'}
    for f,k in [(PARENT/'model.py','model_sha256'),(PARENT/'native.py','native_sha256'),(PARENT/'evaluator.py','evaluator_sha256'),(Path(helper['prior_path']),'prior_sha256'),(EXT,'extension_sha256')]:
        if sha(f)!=helper[k]: raise ValueError('NNUE/prior helper source closure differs')
    if sha(SEARCH)!=SEARCH_SHA: raise ValueError('fixed BudgetSearch source changed')
    sources={name:sha(HERE/name) for name in ('collector.py','run_collection.py','metadata_factory.py')}
    reg=dict(schema='own-nnue-ownq-collection-registration-v2',status='registered',seed=seed,
      operator_end_epoch=operator_end,original_first_epoch=first,original_deadline_epoch=end,
      core_repo=CORE,core_commit=COMMIT,cpu_core=c['cpu_core'],
      root_pool=dict(path=str(dest),sha256=sha(dest),
        selection_path=str(selection),selection_sha256=pool['source_selection_sha256'],
        teacher_labels_path=str(labels),teacher_labels_sha256=pool['source_teacher_labels_sha256']),
      protected_aliases=dict(path=str(binary),sha256=sha(binary),manifest_path=str(manifest),manifest_sha256=sha(manifest),count=len(vals)),
      parent_candidate=dict(path=str(pp),sha256=parent['sha256'],contract_sha256=parent['contract_sha256']),
      teacher_admission=dict(path=str(PROFILE),sha256=PROFILE_SHA,status=profile['status'],strict_native_loads=admission['strict_native_loads'],proof_fresh_loads=admission['proof_fresh_loads'],fit_fresh_loads=admission['fit_fresh_loads']),
      parent_helpers=helper,search_helper=dict(path=str(SEARCH),sha256=SEARCH_SHA),
      output_path=c['output_path'],search=dict(nodes=8192,qdepth=2,max_depth=8),
      row_limit=1024,root_limit=128,plies_per_root=16,producer_source_sha256=sources,
      root_control_sha256=sha(cp),selection_source_sha256=pool['source_selection_sha256'],
      teacher_label_source_sha256=pool['source_teacher_labels_sha256'],protected_source_roles=pm['source_roles'])
    out=Path(c['registration_output']).resolve()
    if out!=Path(f'/workspace/work/harbichess/continuation-20261007/nnue-own-producer-v2/registrations/{seed}.json'):
        raise ValueError('registration output path is fixed to scratch')
    write_once(out,canon(reg)+b'\n')
    return dict(status='PASS-metadata-only-registration',registration_path=str(out),registration_sha256=sha(out),root_pool_path=str(dest),root_pool_sha256=sha(dest),roots=4096,protected_aliases=len(vals))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--control',required=True,type=Path); print(json.dumps(build(p.parse_args().control),sort_keys=True))
