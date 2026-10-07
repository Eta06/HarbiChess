"""Stage the frozen capsule as exact Git blobs; one POST each, no retries."""
import base64
import json
import subprocess
from pathlib import Path

import raw_capsule_v12a as codec
import release_transport_v12a as release


def main():
    here = Path(__file__).parent
    control = json.loads((here / "parts-control.json").read_bytes())
    root = Path(control["local_parts_directory"])
    for row in control["parts"]:
        release.check_clock(control["clock"])
        raw = (root / f"part{row['index']:02}.bin").read_bytes()
        if len(raw) != row["bytes"] or codec.sha(raw) != row["sha256"]:
            raise ValueError("part identity differs before POST")
        started = here / f"part{row['index']:02}-POST-started.json"
        with started.open("x") as stream:
            json.dump(dict(expected_git_blob_sha1=row["git_blob_sha1"], attempts=1), stream)
        body = root / f"part{row['index']:02}-POST-body.json"
        with body.open("x") as stream:
            json.dump(dict(content=base64.b64encode(raw).decode(), encoding="base64"), stream)
        response = subprocess.run([
            "gh", "api", "--method", "POST", "repos/Eta06/HarbiChess/git/blobs",
            "--input", str(body),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        result = dict(returncode=response.returncode, attempts=1,
            response_sha256=codec.sha(response.stdout), error_details_suppressed=True)
        if response.returncode == 0:
            packet = json.loads(response.stdout)
            if packet["sha"] != row["git_blob_sha1"]:
                raise ValueError("returned Git blob identity differs")
            result["git_blob_sha1"] = packet["sha"]
        with (here / f"part{row['index']:02}-POST-result.json").open("x") as stream:
            json.dump(result, stream)
        if response.returncode:
            raise RuntimeError("ambiguous/failed blob POST; preserve and reconcile; do not repeat")
        print(json.dumps(dict(index=row["index"], status="PASS-single-exact-blob-POST")), flush=True)
    with (here / "all-parts-POSTed.json").open("x") as stream:
        json.dump(dict(status="PASS-Git-blobs-not-yet-public-Release", parts=len(control["parts"])), stream)


if __name__ == "__main__":
    main()
