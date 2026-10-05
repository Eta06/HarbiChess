"""ROOT-only optional exact-file read over existing serialized SSH; default plan only."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path

SSH_HELPER = Path("/workspace/work/harbichess/a100/ssh_colab_access.py")

REMOTE_PROGRAM = r"""
import base64,hashlib,json,os,stat,sys
from pathlib import Path
request=json.loads(base64.b64decode(sys.argv[1]))
roots=(Path('/content/harbichess-fullgame-method2-inputs').resolve(strict=True),
       Path('/content/harbichess-runs').resolve(strict=True))
rows=[];whole=0
for item in request['known']+request['unresolved']:
 p=Path(item['path'])
 if p.is_symlink() or not p.is_file():
  rows.append({'path':str(p),'status':'missing-or-not-regular'});continue
 resolved=p.resolve(strict=True)
 if not any(resolved.is_relative_to(root) for root in roots):
  raise ValueError('outside public evidence roots')
 if p.suffix not in ('.json','.py','.gz','.safetensors','.pt','.bundle'):
  raise ValueError('unreviewed evidence suffix')
 before=p.stat()
 if not stat.S_ISREG(before.st_mode):raise ValueError('not regular')
 h=hashlib.sha256()
 with p.open('rb') as f:
  while block:=f.read(1024*1024):h.update(block)
 digest=h.hexdigest()
 if item.get('expected_sha256') and digest!=item['expected_sha256']:
  raise ValueError('recorded SHA differs')
 record={'path':str(p),'status':'exact-regular-file-read','sha256':digest,'bytes':before.st_size,
         'recorded_sha_matched':bool(item.get('expected_sha256'))}
 # Metadata-only for large native/partial blobs; never unpickle or inspect outcomes.
 if p.suffix in ('.json','.py') and before.st_size<=16*1024*1024:
  raw=p.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('file changed while reviewing')
  whole+=len(raw)
  if whole>64*1024*1024:raise ValueError('bounded evidence capture exceeded')
  record['bytes_base64']=base64.b64encode(raw).decode('ascii')
 after=p.stat()
 old=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
 new=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 if old!=new:
  raise ValueError('mutable evidence during review')
 rows.append(record)
print(json.dumps({'schema':'formal45-exact-file-review-v1','records':rows,'no_outcome_inspection':True}))
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_for(request):
    body = base64.b64encode(json.dumps(request).encode()).decode()
    return "python3 -c " + shlex.quote(REMOTE_PROGRAM) + " " + shlex.quote(body)


def bounded_call(helper, command):
    try:
        return helper.call(command, timeout=120, capture_output=True)
    except Exception:
        raise RuntimeError(
            "Exact evidence transport failed; connection details suppressed"
        ) from None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--request-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ssh-helper-sha256")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    assert sha(a.request) == a.request_sha256
    request = json.loads(a.request.read_text())
    assert request["schema"] == "formal45-evidence-resolution-request-v1"
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    command = command_for(request)
    (a.output / "remote-evidence-command.txt").write_text(command + "\n")
    if not a.execute:
        (a.output / "plan.json").write_text(
            json.dumps(
                {
                    "status": "plan-only-no-SSH",
                    "request_sha256": a.request_sha256,
                    "known_files": len(request["known"]),
                    "unresolved_files": len(request["unresolved"]),
                },
                indent=2,
            )
            + "\n"
        )
        print("Plan only; no SSH executed.")
        return
    assert a.ssh_helper_sha256 and sha(SSH_HELPER) == a.ssh_helper_sha256
    spec = importlib.util.spec_from_file_location("root_existing_serialized_transport", SSH_HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    result = bounded_call(helper, command)
    assert result.returncode == 0, "Exact evidence review failed; connection details suppressed"
    raw = result.stdout if isinstance(result.stdout, bytes) else result.stdout.encode()
    assert len(raw) <= 100 * 1024 * 1024
    response = json.loads(raw)
    expected = {r["path"]: r for r in request["known"] + request["unresolved"]}
    assert len(response["records"]) == len(expected)
    seen = set()
    for index, row in enumerate(response["records"]):
        assert row["path"] in expected and row["path"] not in seen
        seen.add(row["path"])
        if row["status"] != "exact-regular-file-read":
            continue
        pinned = expected[row["path"]].get("expected_sha256")
        assert not pinned or row["sha256"] == pinned
        if "bytes_base64" in row:
            body = base64.b64decode(row.pop("bytes_base64"), validate=True)
            assert len(body) == row["bytes"] and hashlib.sha256(body).hexdigest() == row["sha256"]
            local = a.output / f"file-{index:06d}.bin"
            local.write_bytes(body)
            row["exact_local_copy"] = str(local.resolve())
    # Unknown digests remain pending ROOT byte review, never auto-approved.
    (a.output / "review-response.json").write_text(json.dumps(response, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "read-only-evidence-captured-pending-ROOT-review",
                "files": len(response["records"]),
                "response_sha256": sha(a.output / "review-response.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
