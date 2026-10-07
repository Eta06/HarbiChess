"""ROOT-only actual phases; metadata planning is import-safe and makes no model calls."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SEEDS = (20262905, 20262906)
CORE = "6fcc8b476d25495d1c9c413e55b2c7ba4794013e"
INVENTORY = "dbf7e13d5b406ff30d8d4c13c6ea33d9ebfdaf6abc14d18d46a69f98d8a3f8a7"
FORENSIC = "bc275cf5bc69a66ca9875893c30c097c17b2c0fea4e29a7b6598525950a89fd3"
CAPS = dict(
    zero=600,
    convert=600,
    **{
        "zero-profile": 600,
        "proof": 600,
        "fresh-fit": 1800,
        "trained-profile": 600,
        "known160": 7200,
    },
)
RAM = Path("/dev/shm/harbichess-antisymmetric-procedural-v3")


def ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def stage(root):
    p = root / "antisymmetric-procedural-own-v3-production"
    inv = ref(p / "source-inventory.json")
    if inv["sha256"] != INVENTORY:
        raise ValueError("exact frozen antisym source inventory")
    for name, digest in read(inv["path"])["files"].items():
        if ref(p / name)["sha256"] != digest:
            raise ValueError("frozen source changed: " + name)
    return p


def records(root, phase, seed):
    pair = phase in ("zero-profile", "trained-profile", "known160")
    return root / ("antisymmetric-root-v3-" + phase + "-" + ("pair" if pair else str(seed)))


def output(phase, seed):
    return (
        RAM
        / dict(zero="zero", convert="data", proof="proof", **{"fresh-fit": "fit"})[phase]
        / str(seed)
    )


def completed(root, phase, seed):
    r = read(records(root, phase, seed) / "result.json")
    if r["status"] != "PASS-actual-antisymmetric-" + phase + "-not-strength":
        raise ValueError("required prior actual phase did not PASS: " + phase)
    if not r["first"] < r["finished"] <= r["deadline"]:
        raise ValueError("prior original phase clock")
    for item in r["artifacts"].values():
        if ref(item["path"]) != item:
            raise ValueError("prior artifact changed")
    return r


def zero_binding(root, seed):
    completed(root, "zero", seed)
    p = output("zero", seed)
    return {
        name: ref(p / filename)
        for name, filename in dict(
            contract="contract.json",
            native="native.pt",
            candidate="candidate.pt",
            result="result.json",
        ).items()
    }


def child_binding(root, seed):
    proof = completed(root, "proof", seed)
    fit = completed(root, "fresh-fit", seed)
    return dict(
        candidate=ref(output("fresh-fit", seed) / "whole/candidate.pt"),
        proof_result=proof["artifacts"]["phase_result"],
        proof_contract=proof["artifacts"]["contract"],
        proof_build_seal=proof["artifacts"]["build_seal"],
        fit_result=fit["artifacts"]["phase_result"],
        fit_contract=fit["artifacts"]["contract"],
        fit_build_seal=fit["artifacts"]["build_seal"],
    )


def inputs(root, seed, control):
    """Read and hash actual common closure only. No search/model/optimizer calls."""
    p = stage(root)
    bindings = read(
        root / "antisymmetric-human-prior-own-v2-inference/inference-source-bindings.json"
    )
    common = Path(f"/dev/shm/harbichess-human-randomstarts-forensic-data-v4/{seed}")
    provenance = read(common / "provenance.json")
    old_seal = control / "original-common-conversion-seal.json"
    write(old_seal, provenance["inputs"])
    inventory = ref(root / "forensic-v4-inventory.json")
    if inventory["sha256"] != FORENSIC:
        raise ValueError("complete qualified forensic inventory")
    return dict(
        seed=seed,
        core_repo=str(root.parent / "cpu-additive-source-6fcc8b4"),
        core_commit=CORE,
        prior_helper=bindings["prior_ref"],
        search_helper=ref(p / "search.py"),
        compiled_refs=bindings["compiled_refs"],
        conversion=dict(
            forensic_directory=str(root / "human-randomstarts-own-v4-forensic-native"),
            forensic_inventories=[inventory],
            forensic_converter=ref(
                root / "human-randomstarts-own-v4-forensic-native/convert_forensic_v3.py"
            ),
            common_conversion_seal=ref(old_seal),
            common_dataset=ref(common / "dataset.json"),
            common_provenance=ref(common / "provenance.json"),
            common_result=ref(common / "result.json"),
        ),
    )


def protocol(root, p, phase, first, operator):
    """Fixed original known8/resources; no teacher endpoints copied from template."""
    template_path = root / "MC-known160-runtime/protocol-DRAFT.json"
    template = read(template_path)
    keys = [
        "book_path",
        "book_sha256",
        "stockfish_path",
        "stockfish_sha256",
        "memory_max_bytes",
        "disk_min_free_bytes",
        "reserved_output_bytes",
        "games_per_tournament",
        "opening_pairs",
        "max_plies",
        "search_nodes",
        "quiescence_plies",
        "max_depth",
        "stockfish_nodes",
        "per_tournament_seconds",
        "tasks",
        "total_games",
        "whole_seconds",
        "E8_value_helper",
    ]
    q = {key: template[key] for key in keys}
    q.update(
        schema="antisymmetric-procedural-known160-protocol-v3",
        status="registered",
        match_seeds=list(SEEDS),
        source_commit=CORE,
        core_repo=str(root.parent / "cpu-additive-source-6fcc8b4"),
        cpu_core=None,
        ROOToperator_end_epoch=operator,
        original_search=ref(p / "search.py"),
        prior_helper=ref(Path(template["prior_helper"]["path"])),
        original_e8_value=ref(Path(template["original_mixed_value"]["path"])),
        original_e8_protocol=ref(template_path),
        zeros={},
        children={},
        paired_datasets={},
        models={},
        arena_helper_sha256={f.name: ref(f)["sha256"] for f in p.glob("*.py")},
        original_first_epoch=first,
        original_deadline_epoch=min(first + 7200, operator),
        scope=dict(
            development="known8 paired160, not virgin confirmation",
            teacher_labels_used=False,
            target="SAME accepted procedural1024 ownQ paired-orientation representation arm",
            common_family_bridge=(
                "original functional humanzero generated common data; "
                "not own-family on-policy claim"
            ),
            model_family_bridge="literalzero paired family newAdam/RNG, not original NNUE resume",
            next_generation_supported=False,
        ),
    )
    e8 = template["models"][str(SEEDS[0])]["e8"]
    for seed in SEEDS:
        s = str(seed)
        q["zeros"][s] = zero_binding(root, seed)
        completed(root, "convert", seed)
        q["paired_datasets"][s] = ref(output("convert", seed) / "dataset.json")
        if phase == "trained-profile":
            q["children"][s] = child_binding(root, seed)
        q["models"][s] = dict(parent=q["zeros"][s]["candidate"], e8=e8)
        if phase == "trained-profile":
            q["models"][s]["learned"] = q["children"][s]["candidate"]
    return q


def startticks(pid):
    return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[19]


def owned_tree(pid):
    """Snapshot only this launched CLI's descendants, with PID-reuse identities."""
    entries = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            tail = (proc / "stat").read_text().split(")", 1)[1].split()
            entries[int(proc.name)] = dict(
                pid=int(proc.name), parent=int(tail[1]), startticks=tail[19], state=tail[0]
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    selected = {pid} if pid in entries else set()
    changed = True
    while changed:
        before = len(selected)
        selected.update(p for p, item in entries.items() if item["parent"] in selected)
        changed = len(selected) != before
    return [entries[p] for p in selected]


def terminate_owned(items):
    """No name/argv matching, broad signals or unrelated process ownership."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for item in reversed(items):
            try:
                if startticks(item["pid"]) == item["startticks"]:
                    os.kill(item["pid"], sig)
            except ProcessLookupError:
                pass
            except FileNotFoundError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--cpu-core", type=int, choices=(1, 3), required=True)
    parser.add_argument("--phase", choices=CAPS, required=True)
    a = parser.parse_args()
    os.sched_setaffinity(0, {a.cpu_core})
    root = a.root.resolve(strict=True)
    p = stage(root)
    operator = read(root / "operator-window.json")["operator_end_epoch"]
    if operator != 1791448916.685839:
        raise ValueError("frozen ROOT allocation cap")
    control = records(root, a.phase, a.seed)
    control.mkdir(exist_ok=False)
    logs = RAM / "logs" / control.name
    logs.mkdir(parents=True, exist_ok=False)
    first = time.time()
    deadline = min(first + CAPS[a.phase], operator)
    if a.phase == "known160":
        prior = completed(root, "trained-profile", a.seed)
        q = read(prior["artifacts"]["protocol"]["path"])
        # ONE7200 starts when the paired trained profile protocol is frozen.
        first, deadline = q["original_first_epoch"], q["original_deadline_epoch"]
    if time.time() >= deadline:
        raise TimeoutError("phase allocation expired; no reset")
    result = dict(
        schema="ROOT-antisymmetric-procedural-phase-owner-v1",
        status="running",
        phase=a.phase,
        seed=a.seed,
        cpu_core=a.cpu_core,
        first=first,
        deadline=deadline,
        operator_end_epoch=operator,
        helper=ref(__file__),
        source_inventory=ref(p / "source-inventory.json"),
        operator_window=ref(root / "operator-window.json"),
        commands=[],
        artifacts={},
        same_common_data_rows=1024,
        parent_next_generation_supported=False,
        teacher_labels_used=False,
        model_family_bridge="NEW paired literalzero weights/newAdam/RNG, not NNUE resume",
    )
    write(control / "registration.json", result)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONPATH=str(root.parent / "cpu-additive-source-6fcc8b4/src"),
    )

    def run(name, command):
        out, err = logs / (name + ".stdout"), logs / (name + ".stderr")
        started = time.time()
        with out.open("xb") as stdout, err.open("xb") as stderr:
            child = subprocess.Popen(
                command, stdout=stdout, stderr=stderr, env=env, start_new_session=True
            )
            item = dict(
                name=name,
                command=command,
                pid=child.pid,
                startticks=startticks(child.pid),
                started=started,
                process_group=child.pid,
            )
            result["commands"].append(item)
            write(control / (name + ".owned.json"), item)
            try:
                child.wait(timeout=max(0.001, deadline - time.time()))
            except BaseException:
                item["cleanup_owned_processes"] = owned_tree(child.pid)
                terminate_owned(item["cleanup_owned_processes"])
                child.wait(timeout=5)
                raise
            finally:
                item.update(
                    returncode=child.returncode,
                    finished=time.time(),
                    stdout=ref(out),
                    stderr=ref(err),
                )
        if child.returncode or time.time() >= deadline:
            raise RuntimeError("actual phase child failed/expired: " + name)

    def prepare(mode, inp, phase_out, target):
        input_path = control / (mode + "-inputs.json")
        write(input_path, inp)
        run(
            "prepare-" + mode,
            [
                sys.executable,
                str(p / "prepare.py"),
                "--inputs",
                str(input_path),
                "--mode",
                mode,
                "--first",
                str(first),
                "--deadline",
                str(deadline),
                "--operator-end-epoch",
                str(operator),
                "--cpu-core",
                str(a.cpu_core),
                "--phase-output",
                str(phase_out),
                "--output",
                str(target),
            ],
        )

    try:
        if a.phase in ("zero", "convert", "proof", "fresh-fit"):
            inp = inputs(root, a.seed, control)
            out = output(a.phase, a.seed)
            if out.exists():
                raise FileExistsError("no fresh phase overwrite")
            mode = {"zero": "initialize", "convert": "convert"}.get(a.phase, a.phase)
            if a.phase in ("proof", "fresh-fit"):
                conv = completed(root, "convert", a.seed)
                zp = completed(root, "zero-profile", a.seed)
                inp.update(
                    zero_parent=zero_binding(root, a.seed),
                    conversion_seal=conv["artifacts"]["conversion_seal"],
                    paired_dataset=ref(output("convert", a.seed) / "dataset.json"),
                    paired_provenance=ref(output("convert", a.seed) / "provenance.json"),
                    zero_profile=zp["artifacts"]["profile"],
                    zero_profile_protocol=zp["artifacts"]["protocol"],
                )
                if a.phase == "fresh-fit":
                    proof = completed(root, "proof", a.seed)
                    inp.update(
                        own_proof_result=proof["artifacts"]["phase_result"],
                        own_proof_contract=proof["artifacts"]["contract"],
                    )
            seal = control / "build-seal.json"
            prepare(mode, inp, out, seal)
            if a.phase == "zero":
                run(
                    "typedzero-two-fresh",
                    [
                        sys.executable,
                        str(p / "initialize.py"),
                        "--seal",
                        str(seal),
                        "--output",
                        str(out),
                    ],
                )
                phase_result = read(out / "result.json")
                if (
                    phase_result["status"]
                    != "PASS-typedzero-fullnative-two-fresh-opens-not-playing-strength"
                ):
                    raise ValueError("two actual typedzero opens")
                result["artifacts"].update(
                    phase_result=ref(out / "result.json"), contract=ref(out / "contract.json")
                )
            elif a.phase == "convert":
                run(
                    "paired-common-data",
                    [
                        sys.executable,
                        str(p / "convert.py"),
                        "--seal",
                        str(seal),
                        "--output",
                        str(out),
                    ],
                )
                result["artifacts"].update(
                    conversion_seal=ref(seal),
                    dataset=ref(out / "dataset.json"),
                    provenance=ref(out / "provenance.json"),
                )
            else:
                contract = control / "contract.json"
                run(
                    "contract",
                    [
                        sys.executable,
                        str(p / "contracts.py"),
                        "--seal",
                        str(seal),
                        "--output",
                        str(contract),
                    ],
                )
                inp["contract"] = ref(contract)
                registration = control / "phase-registration.json"
                prepare("registry", inp, out, registration)
                run(
                    "full-native-phase",
                    [sys.executable, str(p / "prove.py"), "--registration", str(registration)],
                )
                phase_result = read(out / "result.json")
                if (
                    phase_result["status"]
                    != "PASS-antisymmetric-fixed-phase-and-fresh-native-loads-not-strength"
                ):
                    raise ValueError("actual complete phase/native opens")
                result["artifacts"].update(
                    build_seal=ref(seal),
                    contract=ref(contract),
                    phase_result=ref(out / "result.json"),
                    phase_registration=ref(registration),
                )
        elif a.phase in ("zero-profile", "trained-profile"):
            q = protocol(root, p, a.phase, first, operator)
            q["cpu_core"] = a.cpu_core
            protocol_path = control / "protocol.json"
            write(protocol_path, q)
            clock = dict(
                first=first,
                deadline=deadline,
                operator_end_epoch=operator,
                cpu_core=a.cpu_core,
                core_repo=q["core_repo"],
                core_commit=CORE,
                protocol=ref(protocol_path),
                helper=ref(p / "qualify_profile.py"),
            )
            clock_path = control / "clock.json"
            write(clock_path, clock)
            profile = control / "profile.json"
            run(
                "actual-profile",
                [
                    sys.executable,
                    str(p / "qualify_profile.py"),
                    "--protocol",
                    str(protocol_path),
                    "--clock",
                    str(clock_path),
                    "--mode",
                    "zero" if a.phase == "zero-profile" else "trained",
                    "--output",
                    str(profile),
                ],
            )
            r = read(profile)
            expected = (
                "PASS-zero24-exact-packets-traces-not-strength"
                if a.phase == "zero-profile"
                else "PASS-trained48-parity-paired-latency-not-strength"
            )
            if r["status"] != expected:
                raise ValueError("complete actual zero/trained profile required")
            result["artifacts"].update(
                protocol=ref(protocol_path), profile=ref(profile), clock=ref(clock_path)
            )
        else:
            source_protocol = Path(prior["artifacts"]["protocol"]["path"])
            study = control / "study"
            study.mkdir()
            for f in p.glob("*.py"):
                (study / f.name).symlink_to(f.resolve())
            write(study / "protocol.json", read(source_protocol))
            if ref(study / "protocol.json")["sha256"] != ref(source_protocol)["sha256"]:
                raise ValueError("profile and arena EXACT same protocol bytes")
            out = RAM / "known160"
            run(
                "actual-known160",
                [
                    sys.executable,
                    str(p / "run_known160.py"),
                    "--checkout",
                    q["core_repo"],
                    "--study",
                    str(study),
                    "--qualification",
                    prior["artifacts"]["profile"]["path"],
                    "--first-epoch",
                    str(first),
                    "--output",
                    str(out),
                ],
            )
            cohort = read(out / "cohort-result.json")
            if cohort["status"] != "completed-games-not-strength" or len(cohort["rows"]) != 10:
                raise ValueError("all fixed160 tournaments must close; partial cohort preserved")
            result["artifacts"].update(
                protocol=ref(study / "protocol.json"), owner_result=ref(out / "cohort-result.json")
            )
        result["status"] = "PASS-actual-antisymmetric-" + a.phase + "-not-strength"
    except BaseException as error:
        result.update(status="FAILED-preserved", error=repr(error))
        raise
    finally:
        result["finished"] = time.time()
        write(control / "result.json", result)


if __name__ == "__main__":
    main()
