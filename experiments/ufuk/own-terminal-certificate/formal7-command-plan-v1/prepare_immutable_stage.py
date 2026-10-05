"""Build immutable formal7 input staging payload; no transport or process execution."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

BASE = "/content/harbichess-fullgame-method2-inputs/formal7-production"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assets(repo, plan, books):
    files = {}
    helpers = repo / "experiments/ufuk/own-terminal-method7"
    for file in helpers.iterdir():
        if file.is_file() and (file.suffix == ".py" or file.name == "protocol-DRAFT.json"):
            files[BASE + "/helpers7/" + file.name] = file
    for name in (
        "books-provenance.json",
        "terminal45-barrier-config.json",
        "curriculum-provenance.json",
        "mc-barrier.json",
    ):
        files[BASE + "/" + name] = plan / name
    # Root-promoted book location must be explicit; caller inventory checks these actual files.
    expected = {
        20261725: "cf3455abea53dd1146aaae4a55593e51ca42981cfdacf80cc2a5e9704df48ef8",
        20261726: "13f4722d04a15b512f770c78c65145f96946d7d58ad30db7a20fa9eeef4bd0b7",
    }
    for seed in (20261725, 20261726):
        file = Path(books[seed])
        assert sha(file) == expected[seed]
        files[BASE + f"/book-{seed}.json"] = file
    return files


def render(files):
    packed = {
        path: {"sha256": sha(file), "base64": base64.b64encode(file.read_bytes()).decode()}
        for path, file in files.items()
    }
    return """import base64,hashlib,json,os,tempfile
from pathlib import Path
assets=json.loads(PACKED_LITERAL)
# Check EVERY existing path before the first write; never overwrite historical inputs.
for target,item in assets.items():
 raw=base64.b64decode(item['base64']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
 path=Path(target)
 if path.exists():assert path.read_bytes()==raw,target
for target,item in assets.items():
 path=Path(target);raw=base64.b64decode(item['base64'])
 if path.exists():continue
 path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix='.immutable-stage-',dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as stream:
   stream.write(raw);stream.flush();os.fsync(stream.fileno())
  os.link(tmp,path)
 finally:Path(tmp).unlink(missing_ok=True)
for target,item in assets.items():
 assert hashlib.sha256(Path(target).read_bytes()).hexdigest()==item['sha256']
print(json.dumps({'status':'staged-formal7-inputs-only','processes_launched':0,
 'file_sha256':{path:item['sha256'] for path,item in assets.items()}}))
""".replace("PACKED_LITERAL", repr(json.dumps(packed)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--plan-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--book-20261725", type=Path, required=True)
    parser.add_argument("--book-20261726", type=Path, required=True)
    args = parser.parse_args()
    files = assets(
        args.repo, args.plan_directory, {20261725: args.book_20261725, 20261726: args.book_20261726}
    )
    with args.output.open("x") as stream:
        stream.write(render(files))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "payload_sha256": sha(args.output),
                "file_count": len(files),
                "remote_actions": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
