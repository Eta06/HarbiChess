"""ROOT-registered producer only. This code does not train or call Stockfish."""
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
import torch
from collector import AliasWriter, collect

END=1791273600

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def load(path, digest, name):
    path=Path(path).resolve()
    if sha(path)!=digest: raise ValueError("pinned helper SHA differs: "+path.name)
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod; spec.loader.exec_module(mod)
    if Path(mod.__file__).resolve()!=path: raise ValueError("helper origin differs")
    return mod

def parent(reg):
    h=reg["parent_helpers"]; d=Path(h["directory"]).resolve()
    sys.path.insert(0,str(d)); sys.modules.pop("model",None); sys.modules.pop("native",None)
    m=load(d/"model.py",h["model_sha256"],"model")
    n=load(d/"native.py",h["native_sha256"],"native")
    prior_mod=load(h["prior_path"],h["prior_sha256"],"ownq_prior")
    ext=load(h["extension_path"],h["extension_sha256"],"_kingbucket16")
    e=load(d/"evaluator.py",h["evaluator_sha256"],"ownq_evaluator")
    row=reg["parent_candidate"]; p=Path(row["path"]).resolve()
    if p != Path(f"/dev/shm/harbichess-NNUE-teacher-fit-v1/{reg['seed']}/whole/candidate.pt"):
        raise ValueError("exact future frozen teacher-parent path required")
    if p.stat().st_size>32*2**20 or sha(p)!=row["sha256"]: raise ValueError("parent file hash/size")
    packet=torch.load(p,map_location="cpu",weights_only=False)
    if set(packet)!={"schema","contract","model"} or packet["schema"]!=m.MODEL_SCHEMA:
        raise ValueError("parent candidate schema")
    c=packet["contract"]
    if (hashlib.sha256(canon(c)).hexdigest()!=row["contract_sha256"]
        or c.get("phase")!="teacher-bootstrap" or c.get("updates")!=256
        or c.get("seed")!=reg["seed"] or c.get("feature_schema")!=m.FEATURE_SCHEMA
        or c.get("prior_helper_sha256")!=h["prior_sha256"]):
        raise ValueError("exact same-seed teacher-once parent contract")
    n.validate_weights(packet["model"])
    prior=e.AuthoritativePrior(prior_mod,prior_mod.ClassicalValue())
    return e.Evaluator(packet["model"],prior=prior,compiled=ext)

def execute(path):
    path=Path(path).resolve()
    reg=json.loads(Path(path).read_bytes())
    if reg.get("schema")!="own-nnue-ownq-collection-v1" or reg.get("status")!="registered":
        raise ValueError("ROOT immutable registration required")
    if reg.get("seed") not in (20262905,20262906) or reg.get("search")!={"nodes":8192,"qdepth":2,"max_depth":8}:
        raise ValueError("fixed seed/search contract")
    if (reg.get("row_limit"),reg.get("root_limit"),reg.get("plies_per_root"))!=(1024,128,16):
        raise ValueError("fixed collection budget")
    for name in ("collector.py","run_collection.py"):
        if sha(Path(__file__).with_name(name))!=reg["producer_source_sha256"][name]:
            raise ValueError("registered producer source differs")
    first,end=reg["original_first_epoch"],reg["original_deadline_epoch"]
    if not first<=time.time()<end<=min(END,first+7200): raise TimeoutError("original 7200s clock")
    core=Path(reg["core_repo"]).resolve()
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=core,text=True).strip()!=reg["core_commit"] or subprocess.check_output(["git","status","--porcelain"],cwd=core,text=True):
        raise ValueError("clean core pin")
    os.sched_setaffinity(0,{reg["cpu_core"]}); torch.set_num_threads(1)
    sys.path.insert(0,str(core/"src"))
    from harbichess.training.cgroup_budget import CgroupMemoryBudget
    budget=CgroupMemoryBudget(15*2**30); root=Path(reg["root_pool"]["path"])
    protected_path=Path(reg["protected_aliases"]["path"]); out=Path(reg["output_path"]).resolve()
    if not out.is_relative_to(Path("/dev/shm/harbichess-ownq-v1")) or out.exists(): raise ValueError("publish-once RAM output")
    if sha(root)!=reg["root_pool"]["sha256"] or root.stat().st_size>8*2**20: raise ValueError("4096-root pool pin/size")
    if (sha(protected_path)!=reg["protected_aliases"]["sha256"] or protected_path.stat().st_size%8
            or protected_path.stat().st_size>256*2**20): raise ValueError("protected projection input")
    import struct
    packed=protected_path.read_bytes(); values=list(struct.unpack(f"<{len(packed)//8}q",packed))
    if values!=sorted(set(values)): raise ValueError("protected alias canonical encoding")
    def guard():
        budget.check()
        if shutil.disk_usage("/workspace").free<256*2**20: raise RuntimeError("workspace disk floor")
        if time.time()>=end: raise TimeoutError("original collection clock")
    guard()
    out.mkdir(parents=True,exist_ok=False); log=(out/"events.jsonl").open("xb"); aliases=AliasWriter(out)
    def emit(event, check=True):
        if event["type"]=="search_row":
            event["row"]["search_alias_ref"]=aliases.append(event["row"].pop("eval_aliases"))
        log.write(canon(event)+b"\n"); log.flush(); os.fsync(log.fileno())
        if check: guard()
    try:
        search=load(reg["search_helper"]["path"],reg["search_helper"]["sha256"],"ownq_search")
        value=parent(reg)
        factory=SimpleNamespace(evaluator=value.nonterminal,
            make=lambda fn:search.BudgetSearch(fn,nodes=8192,quiescence_plies=2,max_depth=8,guard=guard))
        pool=json.loads(root.read_bytes())
        result=collect(pool,seed=reg["seed"],protected=set(values),factory=factory,guard=guard,on_event=emit)
        aliases.close()
        audit_ids=[result["training_row_ids"][i] for i in (0,204,409,614,819,1023)]
        chunks=[{"file":p.name,"bytes":p.stat().st_size,"sha256":sha(p)}
                for p in sorted(out.glob("search-aliases-*.bin"))]
        receipt={"schema":"own-nnue-ownq-collection-receipt-v1","status":result["status"],
            "seed":reg["seed"],"registration_sha256":sha(path),
            "parent_candidate_sha256":reg["parent_candidate"]["sha256"],
            "parent_contract_sha256":reg["parent_candidate"]["contract_sha256"],
            "parent_helper_sha256":reg["parent_helpers"],
            "search_helper_sha256":reg["search_helper"]["sha256"],
            "root_pool_sha256":reg["root_pool"]["sha256"],"protected_aliases_sha256":reg["protected_aliases"]["sha256"],
            "selected_root_order_sha256":result["selected_root_order_sha256"],"train_rows":len(result["training_row_ids"]),
            "all_actor_rows":len(result["rows"]),"starts_considered":result["starts_considered"],
            "training_row_ids":result["training_row_ids"],"periodic_independent_search_rows":audit_ids,
            "games":result["games"],"unique_exposure_aliases":len(result["exposed_piece_aliases"]),
            "search_alias_chunks":chunks,
            "search":reg["search"],"teacher_labels_used":False,"target_source":result["target_source"],
            "original_first_epoch":first,"original_deadline_epoch":end,"finished_epoch":time.time()}
        guard()
        with (out/"receipt.json").open("xb") as f:
            f.write(canon(receipt)+b"\n"); f.flush(); os.fsync(f.fileno())
    except BaseException as exc:
        aliases.close(); emit({"type":"producer-failure","error":repr(exc),"time":time.time()},False); raise
    finally:
        log.flush(); os.fsync(log.fileno()); log.close()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--registration",required=True,type=Path)
    execute(p.parse_args().registration)
