"""Fixed family8 transport with exact eligible-model fetch; no background mirror dependency."""

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path

BASE = "/content/harbichess-fullgame-method2-inputs/formal8-production"
MANIFEST = BASE + "/method8-portable-exchange-manifest.json"
RECEIPT = BASE + "/method8-final-mlx-cpu-parity.json"
SOURCE = "3be5b87db27a0fbde83464e7ea7157f0d9a76ae4"
SSH = Path("/workspace/work/harbichess/a100/ssh_colab_access.py")
SSH_SHA = "ccc3b1bf660abf9d6f08d6a0dfbe11563f9f5cad6494c3d62d7346a8f513a89e"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.is_symlink()
    if path.exists():
        assert path.read_bytes() == data
        return
    fd, name = tempfile.mkstemp(prefix=".own8-transfer-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def validate_packet(packet):
    manifest = packet["manifest"]
    assert manifest["source_commit"] == SOURCE
    assert manifest["qualification_ledger_slot"] == 8 and manifest["fixed_epochs"] == 8
    assert [r["seed"] for r in manifest["seeds"]] == [20261825, 20261826]
    assert set(packet["models"]) == {"20261825", "20261826"}
    rows = []
    for r in manifest["seeds"]:
        expected = (
            f"/content/harbichess-runs/visited-loss-method8-seed-{r['seed']}/run/"
            "checkpoints/epoch-00000008/model.safetensors"
        )
        assert r["candidate"] == expected
        body = base64.b64decode(packet["models"][str(r["seed"])], validate=True)
        assert 0 < len(body) < 16 * 1024**2
        assert sha(body) == r["candidate_sha256"]
        rows.append((expected, body))
    return manifest, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mirror", type=Path, required=True)
    p.add_argument("mode", choices=("read", "publish"))
    p.add_argument("remote")
    p.add_argument("local", nargs="?")
    p.add_argument("sha256", nargs="?")
    a = p.parse_args()
    assert sha(SSH.read_bytes()) == SSH_SHA
    spec = importlib.util.spec_from_file_location("ssh_colab_access", SSH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if a.mode == "read":
        assert a.remote == MANIFEST and a.local is a.sha256 is None
        code = f"""import json,base64,hashlib
from pathlib import Path
p=Path({MANIFEST!r});assert not p.is_symlink();manifest=json.loads(p.read_text())
assert manifest['source_commit']=={SOURCE!r} and manifest['fixed_epochs']==8
assert [r['seed'] for r in manifest['seeds']]==[20261825,20261826]
models={{}}
for r in manifest['seeds']:
 expected=f"/content/harbichess-runs/visited-loss-method8-seed-{{r['seed']}}/run/checkpoints/epoch-00000008/model.safetensors"
 assert r['candidate']==expected
 path=Path(expected);assert not path.is_symlink();raw=path.read_bytes()
 assert 0<len(raw)<16*1024**2 and hashlib.sha256(raw).hexdigest()==r['candidate_sha256']
 models[str(r['seed'])]=base64.b64encode(raw).decode()
print(json.dumps({{'manifest':manifest,'models':models}}))
"""
        response = module.call("python3 -c " + shlex.quote(code), capture_output=True, timeout=45)
        if response.returncode:
            return 1  # Manifest not yet published is an expected polling condition.
        manifest, rows = validate_packet(json.loads(response.stdout))
        for remote, body in rows:
            once(a.mirror / Path(remote).relative_to("/content"), body)
        sys.stdout.write(json.dumps(manifest) + "\n")
    else:
        assert a.remote == RECEIPT
        # Reuse the already-qualified immutable receipt publication/readback mechanism.
        sibling = Path(__file__).parents[1] / "own45-cohort/owner_mlx_transport.py"
        spec = importlib.util.spec_from_file_location("owner45_transport", sibling)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        _, _, _, receipt = old.publish_receipt(module.call, a.remote, a.local, a.sha256)
        sys.stdout.write(json.dumps(receipt) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("family8 owner transport failed; no publication claim", file=sys.stderr)
        raise SystemExit(1) from None
