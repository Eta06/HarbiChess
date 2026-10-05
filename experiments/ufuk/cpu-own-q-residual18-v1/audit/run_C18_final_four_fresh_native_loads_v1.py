import json,subprocess,sys,time,hashlib
from pathlib import Path
b=Path("/workspace/work/harbichess/continuation-20261005");d=b/"ownq-C18-training-v1-registration";r=json.loads(Path(sys.argv[1]).read_text());rows=[];hashes={}
for seed in [20262905,20262906]:
 reg=d/f"{seed}-fit-registration.json";q=json.loads(reg.read_text());out=Path(q["output_dir"]);hashes[str(seed)]=[]
 for name,step in [("fresh0",0),("final64",64)]:
  assert time.time()<r["deadline_epoch"]
  p=out/(name+".native.json.gz");h=hashlib.sha256(p.read_bytes()).hexdigest();cmd=[sys.executable,str(b/"strict_native18_readonly_load_v1.py"),"--registration",str(reg),"--native",str(p),"--sha256",h,"--step",str(step)];done=subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=max(1,r["deadline_epoch"]-time.time()));receipt=json.loads(done.stdout);assert receipt["step"]==step and receipt["native_sha256"]==h;rows.append(receipt);hashes[str(seed)].append(h)
result=dict(schema="own-search-C18-final-four-fresh-loads-v1",status="PASS",source_commit="6fcc8b476d25495d1c9c413e55b2c7ba4794013e",native_sha256_by_seed=hashes,fresh_interpreter_loads=4,rows=rows,finished_epoch=time.time(),first_epoch=r["first_epoch"],deadline_epoch=r["deadline_epoch"],optimizer_steps_run=0);assert result["finished_epoch"]<r["deadline_epoch"];Path(r["result_path"]).write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(dict(status=result["status"],fresh_processes=len(rows),finished_epoch=result["finished_epoch"])))
