import argparse,json,gzip,hashlib,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,"/workspace/work/harbichess/cpu-own-search-reanalysis-18x32-proposal")
from residual_critic18 import ResidualCritic18
from train_residual18 import native_digest
p=argparse.ArgumentParser()
for name in ["native","registration"]:p.add_argument("--"+name,type=Path,required=True)
p.add_argument("--sha256",required=True);p.add_argument("--step",type=int,required=True);a=p.parse_args();r=json.loads(a.registration.read_text());clock=r["clocks"][r["phase"]];assert clock["original_first_epoch"]<=time.time()<clock["original_deadline_epoch"]
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(a.native)==a.sha256
base=Path("/workspace/work/harbichess/cpu-own-search-reanalysis-18x32-proposal");names={"critic18":"residual_critic18.py","features18":"feature_bundle18.py","reanalyze":"reanalyze.py","trainer18":"train_residual18.py","runner18":"training_cli18_v2.py"}
for key,name in names.items():assert sha(base/name)==r["closure"][key]
for path,expected in r["inference_closure"].items():assert sha(path)==expected
assert sha(r["labels_path"])==r["labels_sha256"]
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=r["source_repo"],text=True).strip()==r["source_commit"]
state=json.loads(gzip.decompress(a.native.read_bytes()));contract=state["contract"];assert state["step"]==a.step
for key in ["source_commit","helper_source_commit","helper_hashes","reanalysis_registration_sha256","actor_config_sha256","actor_journal_sha256","inference_closure"]:assert contract[key]==r[key]
assert contract["training_code_sha256"]==r["closure"] and contract["labels_sha256"]==r["labels_sha256"]
proofdir=Path(r["output_dir"]) if r["phase"]=="proof" else Path(r["proof_result_path"]).parent
groups=json.loads(gzip.decompress((proofdir/"training-groups.json.gz").read_bytes()));can=lambda x:json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode();assert hashlib.sha256(can(groups)).hexdigest()==contract["training_dataset_sha256"]
m=ResidualCritic18(seed=990001,contract=contract,state=state);assert native_digest(state)==native_digest(m.native());assert time.time()<clock["original_deadline_epoch"]
print(json.dumps(dict(status="PASS-fresh-process-full-native-Adam-RNG-data-source-load-no-SGD",native_path=str(a.native),native_sha256=a.sha256,step=a.step,registration_sha256=sha(a.registration),state_digest=native_digest(state),first_epoch=clock["original_first_epoch"],deadline_epoch=clock["original_deadline_epoch"],finished_epoch=time.time(),optimizer_steps_run=0)))
