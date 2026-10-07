"""ROOT-only owned generational TDLeaf execution; frozen H is never edited."""

import argparse
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

END = 1791448916.685839
SEEDS = (20262905, 20262906)
STAGE_INVENTORY_SHA = "d9487eaa9bcd8dbb83d8eae7dbd6996aab33e1e307d035d2650d4b5b58e0d494"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha(path))


def pinned(r):
    p = Path(r["path"])
    if p.is_symlink() or not p.is_file() or sha(p) != r["sha256"]:
        raise ValueError("exact immutable regular-file binding")
    return p


def read(r):
    return json.loads(pinned(r).read_bytes())


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    )
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with tmp.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def proc_identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rindex(")") + 2 :].split()
        return dict(
            pid=int(pid),
            startticks=int(fields[19]),
            ppid=int(fields[1]),
            pgid=int(fields[2]),
            state=fields[0],
        )
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_owner(identity):
    now = proc_identity(identity["pid"])
    return now is not None and now["startticks"] == identity["startticks"] and now["state"] != "Z"


def descendants(root, tracked):
    if root is not None:
        tracked[root["pid"]] = root
    current = {}
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            x = proc_identity(int(p.name))
            if x:
                current[x["pid"]] = x
    parents = {p for p, r in tracked.items() if same_owner(r)}
    while True:
        more = {p: x for p, x in current.items() if x["ppid"] in parents and p not in parents}
        if not more:
            break
        tracked.update(more)
        parents.update(more)
    return tracked


def stop_owned(tracked):
    """Exact identities/pidfds only, no group/name signalling or old owners."""
    for signum in (signal.SIGTERM, signal.SIGKILL):
        for record in list(tracked.values())[::-1]:
            if not same_owner(record):
                continue
            try:
                fd = os.pidfd_open(record["pid"])
                try:
                    if same_owner(record):
                        signal.pidfd_send_signal(fd, signum)
                finally:
                    os.close(fd)
            except ProcessLookupError:
                pass
        until = time.monotonic() + 2
        while time.monotonic() < until and any(same_owner(r) for r in tracked.values()):
            time.sleep(0.05)
    if any(same_owner(r) for r in tracked.values()):
        raise RuntimeError("owned child did not terminate")


STRENGTH_SCOPE = {
    "endpoint_generation": 3,
    "required_parent_baseline": "literalzero-generation0",
    "required_current_parent": "CURRENT-generation2",
    "development_arms": 7,
    "development_games": 224,
    "formal_arms": 7,
    "formal_games": 1344,
    "formal_lower_bounds": 12,
    "formal_per_bound_alpha": 0.0005208333333333333,
    "formal_e_threshold": 1920,
    "formal_joint_alpha": 0.00625,
    "historical_plus_new_bound": 0.05625,
    "numerical_gates": "unchanged-no-relaxation",
    "formal_campaign": "SAME-ONE-undrawn-root-draw-no-reroll",
}


def validate_strength_scope(scope):
    if scope != STRENGTH_SCOPE:
        raise ValueError("prospective REQUIRED g0/g2 endpoint and exact family allocation")


def setup(config, allow_draft=False):
    if config["schema"] != "ROOT-generational-TDLeaf-control-config-v2-required-parents" or config[
        "status"
    ] not in (["registered", "DRAFT-source-preflight-only"] if allow_draft else ["registered"]):
        raise ValueError("ROOT prospective config required")
    if config["operator_end_epoch"] != END or config["seeds"] != list(SEEDS):
        raise ValueError("fixed technical END/seeds")
    required = {
        "control.py",
        "collect_entry.py",
        "seal_plan.py",
        "learn.py",
        "train_entry.py",
        "generation_entry.py",
        "run_all.py",
    }
    if not required <= set(config["control_source_sha256"]):
        raise ValueError("complete seven-file integration execution closure")
    for name, digest in config["control_source_sha256"].items():
        if Path(name).name != name or sha(Path(__file__).with_name(name)) != digest:
            raise ValueError("exact integration controller source closure")
    h = Path(config["helper_directory"]).resolve()
    inv = read(config["helper_inventory"])
    if config["helper_inventory"]["sha256"] != STAGE_INVENTORY_SHA:
        raise ValueError("frozen original generational H inventory")
    for name, digest in inv["files"].items():
        if sha(h / name) != digest:
            raise ValueError("frozen H bytes differ " + name)
    for name in ("parent_bridge", "zero_parent", "generation_plan", "root_bank"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(h))
    bridge = importlib.import_module("parent_bridge")
    core = Path(config["core_repo"])
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=core, text=True).strip()
        != config["core_commit"]
    ):
        raise ValueError("exact clean core commit")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=core, text=True):
        raise ValueError("clean core source")
    sys.path.insert(0, str(core / "src"))
    for seed in SEEDS:
        seal = read(config["zero_parent_seals"][str(seed)])
        contract = bridge.validate_metadata(seal)
        if (
            contract["phase"] != "human-prior-zero-residual-init-v1"
            or contract["generation"] != 0
            or contract["seed"] != seed
            or contract["updates"] != 0
        ):
            raise ValueError("actual original literalzero g0 only; no teacher/own128 masquerade")
        if contract["search_helper"] != config["search_helper"]:
            raise ValueError("same exact de53 search")
        directory = Path(config["parent_helpers"]["directory"])
        pairs = [
            (directory / "model.py", "model_sha256"),
            (directory / "native.py", "native_sha256"),
            (directory / "evaluator.py", "evaluator_sha256"),
            (Path(config["parent_helpers"]["prior_path"]), "prior_sha256"),
            (Path(config["parent_helpers"]["extension_path"]), "extension_sha256"),
        ]
        for path, key in pairs:
            digest = config["parent_helpers"][key]
            if sha(path) != digest or contract["inference_source_sha256"].get(str(path)) != digest:
                raise ValueError("complete actual zero-parent inference closure")
    pinned(config["protected_aliases"])
    pinned(config["search_helper"])
    if (
        config["search_helper"]["sha256"]
        != "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670"
    ):
        raise ValueError("original de53 control")
    validate_strength_scope(config["strength_scope"])
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    budget = CgroupMemoryBudget(15 * 2**30)
    return h, bridge, budget


class Owner:
    def __init__(self, config, root):
        self.config = config
        self.root = Path(root)
        self.commands = []
        self.h, self.bridge, self.budget = setup(config)
        self.root.mkdir(parents=True, exist_ok=False)
        publish(
            self.root / "owner.json",
            dict(
                schema="ROOT-owned-generational-controller-v1",
                config=config,
                control_helper=ref(__file__),
                pid=os.getpid(),
                identity=proc_identity(os.getpid()),
                observed_first=time.time(),
            ),
        )

    def guard(self, deadline):
        import shutil

        self.budget.check()
        if time.time() >= deadline or time.time() >= END:
            raise TimeoutError("original phase/technical deadline exhausted")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("workspace floor256MiB")

    def phase(self, name, seconds, build):
        first = time.time()
        end = min(first + seconds, END)
        self.guard(end)
        phase = self.root / name
        phase.mkdir(exist_ok=False)
        tracked = {}
        proc = None
        record = dict(
            name=name,
            first=first,
            deadline=end,
            helper_sha256=sha(__file__),
            status="registered-before-process",
            new_clock_not_resume=True,
        )
        try:
            cmd = build(phase, first, end)
            record["command"] = cmd
            publish(phase / "command.json", record)
            with (phase / "stdout.log").open("xb") as out, (phase / "stderr.log").open("xb") as err:
                proc = subprocess.Popen(
                    cmd,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                    env={
                        **os.environ,
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    },
                )
                identity = proc_identity(proc.pid)
                record["identity"] = identity
                publish(
                    phase / "process.json",
                    dict(pid=proc.pid, identity=identity, first=first, deadline=end),
                )
                while proc.poll() is None:
                    descendants(identity, tracked)
                    self.guard(end)
                    time.sleep(0.1)
                descendants(identity, tracked)
                if proc.returncode:
                    raise RuntimeError("owned child exit " + str(proc.returncode))
            self.guard(end)
            record.update(status="PASS-owned-command-not-strength", returncode=proc.returncode)
        except BaseException as e:
            record.update(status="FAILED-preserved", error=repr(e))
            if proc:
                descendants(record.get("identity"), tracked)
                stop_owned(tracked)
                proc.wait()
            raise
        finally:
            record.update(
                finished=time.time(),
                tracked_owned=list(tracked.values()),
                stdout_sha256=sha(phase / "stdout.log")
                if (phase / "stdout.log").exists()
                else None,
                stderr_sha256=sha(phase / "stderr.log")
                if (phase / "stderr.log").exists()
                else None,
            )
            publish(phase / "result.json", record)
            self.commands.append(record)
        return phase

    def banks(self, seed):
        import root_bank

        results = {}
        for generation in (1, 2, 3):
            bankseed = seed + 1000003 * generation

            def build(p, first, end, bankseed=bankseed):
                reg = dict(
                    schema="human-procedural-root-bank-registration-v2",
                    status="registered",
                    seed=bankseed,
                    first=first,
                    deadline=end,
                    operator_end_epoch=END,
                    recipe=root_bank.RECIPE,
                    generator_sha256=sha(self.h / "root_bank.py"),
                    protected_aliases=self.config["protected_aliases"],
                    teacher_labels_used=False,
                )
                publish(p / "registration.json", reg)
                return [
                    sys.executable,
                    str(self.h / "root_bank.py"),
                    "--registration",
                    str(p / "registration.json"),
                    "--output",
                    str(p / "bank"),
                ]

            p = self.phase(f"bank-{seed}-g{generation}", 600, build)
            receipt = ref(p / "bank/receipt.json")
            bank = read(receipt)
            if (
                bank["status"] != "PASS-rule-only-procedural-root-bank"
                or bank["accepted_count"] != 4096
            ):
                raise ValueError("complete actual bank only")
            results[str(generation)] = receipt
        publish(self.root / f"banks-{seed}.json", results)
        return results

    def parent_admission(self, seed, generation, parent_seal):
        def build(p, first, end):
            publish(p / "seal.json", parent_seal)
            clock = dict(
                schema="human-prior-own-parent-readonly-admission-clock-v1",
                status="registered",
                first=first,
                deadline=end,
                operator_end_epoch=END,
                helper_sha256=sha(self.h / "admit_parent.py"),
                seal_sha256=sha(p / "seal.json"),
            )
            publish(p / "clock.json", clock)
            return [
                sys.executable,
                str(self.h / "admit_parent.py"),
                "--seal",
                str(p / "seal.json"),
                "--clock",
                str(p / "clock.json"),
                "--output",
                str(p / "admission.json"),
            ]

        p = self.phase(f"admit-{seed}-g{generation}", 600, build)
        self.bridge.validate_admission_result(ref(p / "seal.json"), ref(p / "admission.json"))
        return ref(p / "seal.json"), ref(p / "admission.json")

    def generation(self, seed, generation, plan_ref, parent_seal):
        if generation not in (1, 2, 3):
            raise ValueError("fixed three generations only")
        os.sched_setaffinity(0, {self.config["cpu_cores"][str(seed)]})
        seal, admission = self.parent_admission(seed, generation, parent_seal)
        import generation_plan

        plan = generation_plan.validate_plan(plan_ref, seed)

        def build(p, first, end):
            spec = dict(
                schema="procedural-generational-tdleaf-collection-build-seal-v2",
                status="registered",
                first=first,
                deadline=end,
                operator_end_epoch=END,
                cpu_core=self.config["cpu_cores"][str(seed)],
                parent_admission_seal=seal,
                parent_admission_result=admission,
                parent_helpers=self.config["parent_helpers"],
                search_helper=self.config["search_helper"],
                protected_aliases=self.config["protected_aliases"],
                generation_plan=plan_ref,
                procedural_bank_receipt=plan["banks"][str(seed)][str(generation)],
                root_pool_output=str(p / "root-pool.json"),
            )
            publish(p / "seal.json", spec)
            # Dedicated integration command retains one first/deadline across factory+actor.
            return [
                sys.executable,
                str(Path(__file__).with_name("collect_entry.py")),
                "--helper-directory",
                str(self.h),
                "--seal",
                str(p / "seal.json"),
                "--registration",
                str(p / "registration.json"),
            ]

        p = self.phase(f"collect-{seed}-g{generation}", 10800, build)
        reg = read(ref(p / "registration.json"))
        receipt = ref(Path(reg["output_path"]) / "receipt.json")
        if read(receipt)["status"] != "PASS-exact-row-budget":
            raise ValueError("fixed2048 failed")
        publish(
            self.root / f"collection-{seed}-g{generation}.json",
            dict(
                registration=ref(p / "registration.json"),
                receipt=receipt,
                parent_admission_seal=seal,
                parent_admission_result=admission,
            ),
        )
        return p


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--mode", choices=["preflight", "banks", "generation"], required=True)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--generation", type=int, choices=[1, 2, 3])
    p.add_argument("--plan", type=Path)
    p.add_argument("--plan-sha256")
    p.add_argument("--parent-seal", type=Path)
    p.add_argument("--parent-seal-sha256")
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    cfg = read(dict(path=str(a.config), sha256=a.config_sha256))
    if a.mode == "preflight":
        setup(cfg, allow_draft=True)
        print(json.dumps(dict(status="PASS-readonly-real-zero-metadata-no-model-forward")))
        return
    if not a.execute or a.seed is None or a.output is None:
        p.error("ROOT --execute/--seed/--output required")
    owner = Owner(cfg, a.output)
    if a.mode == "banks":
        os.sched_setaffinity(0, {cfg["cpu_cores"][str(a.seed)]})
        owner.banks(a.seed)
    else:
        if a.generation is None or a.plan is None or a.plan_sha256 is None:
            p.error("fixed generation/plan required")
        parent = (
            read(dict(path=str(a.parent_seal), sha256=a.parent_seal_sha256))
            if a.parent_seal
            else read(cfg["zero_parent_seals"][str(a.seed)])
        )
        owner.generation(a.seed, a.generation, dict(path=str(a.plan), sha256=a.plan_sha256), parent)


if __name__ == "__main__":
    main()
