"""One owned delivery dispatch; then ROOT anonymous full-part and native-byte audit."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import parts_receiver_v11 as parts
import raw_capsule_v11 as codec
import release_transport_v11 as release


def gh(arguments, timeout=30):
    result = subprocess.run(["gh", "api", *arguments], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode:
        raise RuntimeError("GitHub request failed; details suppressed")
    return json.loads(result.stdout) if result.stdout else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    reg = json.loads(args.registration.read_bytes())
    here = Path(__file__).parent
    control_raw = (here / "parts-control.json").read_bytes()
    if codec.sha(control_raw) != reg["parts_control_sha256"]:
        raise ValueError("registered exact parts control")
    control = json.loads(control_raw)
    manifest_raw = (here / "manifest.json").read_bytes()
    parts.validate(control, manifest_raw)
    manifest = json.loads(manifest_raw)
    release.check_clock(control["clock"])
    title = "NNUE V11 parts / " + reg["parts_control_sha256"][:12]
    started = here / "dispatch-started.json"
    with started.open("x") as stream:
        json.dump(dict(epoch=time.time(), attempts=1, expected_head=reg["head"], title=title), stream)
    body = Path(control["local_parts_directory"]) / "dispatch-body.json"
    with body.open("x") as stream:
        json.dump(dict(ref="main", inputs=dict(parts_control_sha=reg["parts_control_sha256"],
            control_prefix=reg["parts_control_sha256"][:12])), stream)
    ambiguous = False
    endpoint = "repos/Eta06/HarbiChess/actions/workflows/nnue-own-state-parts-v11.yml"
    try:
        gh(["--method", "POST", endpoint + "/dispatches", "--input", str(body)])
    except Exception:
        ambiguous = True  # Never repeat POST. Reconcile only our exact owned run.
    run = None
    while True:
        release.check_clock(control["clock"])
        runs = gh([endpoint + "/runs?event=workflow_dispatch&per_page=30"])["workflow_runs"]
        owned = [x for x in runs if x["head_sha"] == reg["head"] and x["display_title"] == title]
        if len(owned) > 1:
            raise RuntimeError("duplicate owned workflow runs; no retry")
        if owned:
            run = owned[0]
            if run["status"] == "completed":
                break
        print(json.dumps(dict(status="waiting-owned-delivery", id=run["id"] if run else None)), flush=True)
        time.sleep(15)
    if run["conclusion"] != "success":
        raise RuntimeError("owned workflow did not succeed; preserve failure without redispatch")
    transport = release.Transport(None, control["clock"], dict(id=404068972,
        tag="ufuk-cpu-ownplay-state-20261005", target_commitish="31a18e981e27e696fcca69c7e62210e7f0e89830",
        initial_assets=1))
    chunks = []
    proofs = []
    for row in control["parts"]:
        release.check_clock(control["clock"])
        raw = transport.download(row["asset_name"], row["bytes"])
        if codec.sha(raw) != row["sha256"]:
            raise ValueError("anonymous complete part SHA mismatch")
        chunks.append(raw)
        proofs.append(dict(name=row["asset_name"], bytes=len(raw), sha256=codec.sha(raw),
            public_url=release.public_url(control["release_tag"], row["asset_name"])))
    joined = b"".join(chunks)
    if len(joined) != control["full_capsule_bytes"] or codec.sha(joined) != control["full_capsule_sha256"]:
        raise ValueError("anonymous joined exact capsule differs")
    originals = codec.verify(joined, manifest)
    release.check_clock(control["clock"])
    result = dict(status="PASS-ROOT-anonymous-all-parts-and-original-bytes-not-strength",
        workflow_id=run["id"], workflow_url=run["html_url"], head_sha=reg["head"],
        POST_attempts=1, ambiguous_POST_reconciled=ambiguous, parts=proofs, originals=originals,
        first=control["clock"]["started_epoch"], deadline=control["clock"]["deadline_epoch"],
        finished=time.time(), full_capsule_sha256=codec.sha(joined),
        manifest_sha256=codec.sha(manifest_raw), original_files=len(originals),
        native_CPU_ABI_restore_claimed=False, strength_success=False)
    with (here / "ROOT-public-readback.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ("parts", "originals")}), flush=True)


if __name__ == "__main__":
    main()
