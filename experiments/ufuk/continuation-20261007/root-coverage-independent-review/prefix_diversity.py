"""Empirical prefix diversity from sealed pool, deterministic existing schedules only."""
import collections
import hashlib
import json
import os
import struct
from pathlib import Path

import chess

BASE=Path('/workspace/work/harbichess/continuation-20261007')
OUT=BASE/'root-coverage-independent-review'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stats(rows):
    out={'rows':len(rows)}
    for n in [4,8,16]:
        eligible=[tuple(r['prefix_uci'][:n]) for r in rows if len(r['prefix_uci'])>=n]
        count=collections.Counter(eligible)
        out[str(n)]=dict(complete_prefix_rows=len(eligible),too_short=len(rows)-len(eligible),
            unique_complete_prefixes=len(count),
            top8=[dict(uci=list(k),rows=v,fraction_of_all_rows=v/len(rows))
                  for k,v in sorted(count.items(),key=lambda x:(-x[1],x[0]))[:8]])
    return out


def main():
    os.nice(15)
    results=[]
    for seed in [20262905,20262906]:
        p=BASE/f'nnue-own-producer-v2/input-pools/{seed}.json'
        rows=json.loads(p.read_bytes())['rows']
        receipt=json.loads(Path(f'/dev/shm/harbichess-ownq-v2/{seed}/receipt.json').read_bytes())
        assert receipt['status']=='PASS-exact-row-budget'
        q=json.loads((BASE/'variant-known160-v1/protocol.json').read_bytes())
        reg=json.loads(Path(q['children'][str(seed)]['collection_registration']['path']).read_bytes())
        assert sha(p)==receipt['root_pool_sha256']
        protected=Path(reg['protected_aliases']['path']);raw=protected.read_bytes()
        assert sha(protected)==reg['protected_aliases']['sha256']
        blocked=set(struct.unpack(f'<{len(raw)//8}q',raw))
        usable=[]
        for r in rows:
            b=chess.Board(r['root_fen'])
            for u in r['prefix_uci']:
                move=chess.Move.from_uci(u);assert b.is_legal(move);b.push(move)
            raw_alias=min(b.board_fen(),b.mirror().board_fen()).encode()
            alias=int.from_bytes(hashlib.sha256(raw_alias).digest()[:8],'little',signed=True)
            if alias not in blocked:usable.append(r)
        chosen=sorted(usable,key=lambda r:hashlib.sha256(
            f"own-nnue-ownq-start-v1|{seed}|{r['root_id']}".encode()).digest())[:128]
        order=hashlib.sha256('\n'.join(r['root_id'] for r in chosen).encode()).hexdigest()
        assert order==receipt['selected_root_order_sha256']
        results.append(dict(seed=seed,pool_sha256=sha(p),protected_sha256=sha(protected),
            fixed128_order_sha256=order,pool=stats(rows),fixed128=stats(chosen)))
    result=dict(schema='independent-sealed-pool-prefix-diversity-v1',seeds=results,
        actual_model_reads=0,searches=0,root_selection_changes=0,
        caveat='Full prefixes count human move sequences, not ECO families or statistical independence; deterministic schedule reconstruction does not select new roots.')
    (OUT/'prefix-diversity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(sha256=sha(OUT/'prefix-diversity.json'),seeds=[dict(seed=r['seed'],
        pool_unique={n:r['pool'][n]['unique_complete_prefixes'] for n in ['4','8','16']},
        fixed128_unique={n:r['fixed128'][n]['unique_complete_prefixes'] for n in ['4','8','16']},
        fixed128_top4=r['fixed128']['4']['top8']) for r in results])))


if __name__=='__main__':main()
