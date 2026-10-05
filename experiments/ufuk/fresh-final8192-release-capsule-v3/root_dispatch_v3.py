"""ROOT-only serial direct Release recovery. No execution on import; no Azure API."""

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

FIRST = 1791226974.7925532
END = 1791228774.7925532
STAGE = Path("/workspace/HarbiChess/experiments/ufuk/fresh-final8192-release-capsule-v3")
REPO = "Eta06/HarbiChess"
WORKFLOW = "fresh-final8192-release-capsule-v3.yml"
OUTPUT_LIMIT = 256 * 1024
RAM_LIMIT = 8 * 1024**2
DISK_FLOOR = 256 * 1024**2


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def owned_title(mode, sha, ordinal):
    return f"Final8192 v3 {mode} / {sha[:12]} / {ordinal}"


def select_owned(batch, prior, title):
    matches = [
        r
        for r in batch
        if r["id"] not in prior
        and r["display_title"] == title
        and r["event"] == "workflow_dispatch"
    ]
    if len(matches) > 1:
        raise RuntimeError("ambiguous-owned-dispatch")
    return matches[0] if matches else None


def clock_guard(control, now=None):
    now = time.time() if now is None else now
    if (
        control.get("started_epoch") != FIRST
        or control.get("deadline_epoch") != END
        or not FIRST <= now < END
    ):
        raise TimeoutError("original-1800-clock-expired-or-changed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--workflow-sha256", required=True)
    ap.add_argument("--receiver-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ram", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(STAGE))
    spec = importlib.util.spec_from_file_location(
        "direct_release_v3", STAGE / "artifact_capsule.py"
    )
    receiver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(receiver)
    from launcher_inputs import dispatch_inputs

    control = json.loads((STAGE / "transport-control-PENDING.json").read_bytes())
    clock_guard(control)
    receiver.release.check_clock(control)
    manifest = STAGE / "manifest-8192-DRAFT.json"
    raw = manifest.read_bytes()
    if (
        digest(manifest) != args.manifest_sha256
        or digest(STAGE / "artifact_capsule.py") != args.receiver_sha256
    ):
        raise ValueError("approved-manifest-receiver-pin")
    m = json.loads(raw)
    receiver.validate(m)
    workflow = STAGE.parents[2] / ".github/workflows" / WORKFLOW
    if digest(workflow) != args.workflow_sha256 or m["workflow_sha256"] != args.workflow_sha256:
        raise ValueError("approved-workflow-pin")
    if m["receiver_sha256"] != args.receiver_sha256:
        raise ValueError("manifest-receiver-binding")
    args.output.mkdir(exist_ok=False)
    args.ram.mkdir(exist_ok=False)
    rows = []
    result = dict(
        schema="fresh-final8192-direct-release-recovery-owner-v3",
        status="failed-preserved",
        original_first_epoch=FIRST,
        original_deadline_epoch=END,
        actual_controller_started_epoch=time.time(),
        manifest_sha256=args.manifest_sha256,
        workflow_sha256=args.workflow_sha256,
        receiver_sha256=args.receiver_sha256,
        controller_sha256=digest(Path(__file__)),
        old_failed_v2_run_retained=37360596243,
        old_v2_artifact_retained=11365929280,
        network_route="officialGitHubAPI+anonymousPUBLICReleaseONLY",
        rows=rows,
    )

    def guard():
        clock_guard(control)
        receiver.release.check_clock(control)
        if shutil.disk_usage(args.output.parent).free < DISK_FLOOR:
            raise RuntimeError("disk256MiBfloor")
        if sum(p.stat().st_size for p in args.output.iterdir() if p.is_file()) > OUTPUT_LIMIT:
            raise RuntimeError("metadata256KiBceiling")
        if sum(p.stat().st_size for p in args.ram.iterdir() if p.is_file()) > RAM_LIMIT:
            raise RuntimeError("opaqueRAM8MiBceiling")

    def gh(arguments, request=None):
        guard()
        command = ["gh", "api", *arguments]
        if request is not None:
            command += ["--input", str(request)]
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=min(30, END - time.time()),
            check=False,
        )
        if proc.returncode or len(proc.stdout) > 1024**2:
            raise RuntimeError("github-api-failed-or-size-bound-no-raw-details")
        guard()
        return json.loads(proc.stdout) if proc.stdout.strip() else None

    runs_path = f"repos/{REPO}/actions/workflows/{WORKFLOW}/runs?per_page=100"

    def dispatch(mode, data, ordinal=0):
        guard()
        payload = (
            ""
            if mode == "aggregate"
            else base64.b64encode(
                data[ordinal * receiver.CHUNK : (ordinal + 1) * receiver.CHUNK]
            ).decode()
        )
        request = dispatch_inputs(raw, args.manifest_sha256, mode, ordinal=ordinal, payload=payload)
        path = args.ram / f"{mode}-{ordinal}.request.json"
        with path.open("xb") as stream:
            stream.write(request)
        os.chmod(path, 0o600)
        prior = {r["id"] for r in gh([runs_path])["workflow_runs"]}
        gh(["--method", "POST", f"repos/{REPO}/actions/workflows/{WORKFLOW}/dispatches"], path)
        title = owned_title(mode, args.manifest_sha256, ordinal)
        ident = None
        observed = time.time()
        while True:
            guard()
            run = (
                select_owned(gh([runs_path])["workflow_runs"], prior, title)
                if ident is None
                else gh([f"repos/{REPO}/actions/runs/{ident}"])
            )
            if run:
                ident = run["id"]
                if (
                    run["display_title"] != title
                    or run["event"] != "workflow_dispatch"
                    or run.get("path") != f".github/workflows/{WORKFLOW}"
                    or run["repository"]["full_name"] != REPO
                ):
                    raise RuntimeError("owned-run-binding")
                if run["status"] == "completed":
                    record = {
                        k: run[k]
                        for k in (
                            "id",
                            "status",
                            "conclusion",
                            "head_sha",
                            "html_url",
                            "display_title",
                            "created_at",
                            "updated_at",
                        )
                    }
                    record.update(
                        mode=mode,
                        ordinal=ordinal,
                        manifest_sha256=args.manifest_sha256,
                        original_deadline_epoch=END,
                    )
                    with (args.output / f"{mode}-{ordinal}.result.json").open("x") as stream:
                        json.dump(record, stream, sort_keys=True, indent=2)
                    rows.append(record)
                    if run["conclusion"] != "success":
                        raise RuntimeError("owned-workflow-not-success")
                    return ident
            if time.time() - observed >= 600:
                raise TimeoutError("single-run600-observation-cap-within-original-clock")
            time.sleep(min(5, max(0, END - time.time())))

    try:
        guard()
        originals = {r["path"]: Path(r["original_path"]).read_bytes() for r in m["files"]}
        for row in m["files"]:
            if (
                len(originals[row["path"]]) != row["bytes"]
                or receiver.sha(originals[row["path"]]) != row["sha256"]
            ):
                raise ValueError("immutable-local-original-changed")
        data = receiver.capsule(originals)
        receiver.verify(data, m)
        if len(data) > 4 * 1024**2:
            raise ValueError("capsule4MiBbound")
        # Anonymous-only transport on PRIMARY: no token read, output, or authenticated fallback.
        public = receiver.release.Transport("", control)
        dispatch("chunk", data, 0)
        part = public.download(receiver.chunk_name(m, 0), m["chunks"][0]["bytes"])
        if receiver.sha(part) != m["chunks"][0]["sha256"]:
            raise ValueError("public-chunk0-full-body-sha")
        guard()
        with (args.output / "chunk0-public-smoke.json").open("x") as stream:
            json.dump(
                dict(
                    status="PASS-one-public-chunk-not-complete-backup",
                    sha256=receiver.sha(part),
                    bytes=len(part),
                    finished_epoch=time.time(),
                ),
                stream,
            )
        for ordinal in range(1, len(m["chunks"])):
            dispatch("chunk", data, ordinal)
        dispatch("aggregate", data)
        downloaded = public.download(f"sha256-{m['capsule_sha256']}.tar.gz", m["capsule_bytes"])
        receiver.verify(downloaded, m)  # Complete anonymous body + ALL4original extractedSHAs.
        guard()
        for row in m["files"]:
            if digest(Path(row["original_path"])) != row["sha256"]:
                raise ValueError("original-input-changed-after-publication")
        result.update(
            status="PASS-original-clock-21-publicchunks-capsule-four-original-SHAs",
            anonymous_full_capsule_sha256=receiver.sha(downloaded),
            four_original_byte_SHA_verified=True,
        )
    except BaseException as error:
        result["error_code"] = type(error).__name__
        result["error"] = (
            str(error)
            if isinstance(error, ValueError | RuntimeError | TimeoutError)
            else "details-suppressed"
        )
    finally:
        result["finished_epoch"] = time.time()
        if result["finished_epoch"] >= END:
            result["status"] = "INCOMPLETE-original-transport-deadline-exhausted"
        with (args.output / "cohort-result.json").open("x") as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
        print(json.dumps({k: result[k] for k in ("status", "finished_epoch", "manifest_sha256")}))


if __name__ == "__main__":
    main()
