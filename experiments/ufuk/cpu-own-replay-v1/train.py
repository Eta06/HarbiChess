"""Bounded CPU-only fitting of verified own terminal/search TRAIN experience.

This has a separate full training-only native format, not an online actor resume.
No engine/teacher query, new game, strength evaluation or model promotion occurs.
"""

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from harbichess.backends.torch_network import load_weights, save_weights, sha256
from harbichess.chess.rules import PythonChessRules
from harbichess.selfplay.online_epoch import deserialize_policy_epoch
from harbichess.training.cgroup_budget import CgroupMemoryBudget
from harbichess.training.fullgame_own_targets import GameBalancedSampler, build_fullgame_targets
from harbichess.training.torch_array_encoder import TorchArrayBoardEncoder
from harbichess.training.torch_fullgame_ppo import _tensor_batch, compile_epoch_features
from harbichess.training.torch_ownsearch_core import search_train_rows
from harbichess.training.torch_search_acting_run import clean_source

SCHEMA = "cpu-own-replay-training-native-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def partition(source_id):
    # Entire opening alias remains in one TRAIN-only internal split across games.
    return int.from_bytes(hashlib.sha256(source_id.encode()).digest()[:8], "big") % 5


def prepare(journals, guard):
    rules = PythonChessRules()
    encoder = TorchArrayBoardEncoder(rules)
    training, validation, policies, receipts = [], [], [], []
    for path in journals:
        guard()
        record = json.loads(gzip.decompress(path.read_bytes()))
        epoch = deserialize_policy_epoch(canonical(record["collection"]))
        own = build_fullgame_targets(rules, epoch, claim_draw=True)
        if len(own.targets) != len(epoch.actions) - record["target_counts"]["excluded_actions"]:
            raise ValueError("own replay known/unknown target counts disagree")
        source = sha256(path)
        for row in own.targets:
            # Keep separate sampled games from different immutable journals.
            target = replace(row, source_id=source + ":" + row.source_id)
            (validation if partition(row.source_id) == 0 else training).append(target)
        for row in search_train_rows(epoch, record["own_search"]):
            if partition(row.source_id) != 0:
                policies.append(replace(row, source_id=source + ":" + row.source_id))
        receipts.append(
            {
                "journal_sha256": source,
                "known_rows": len(own.targets),
                "complete_games": own.complete_games,
                "search_rows": len(record["own_search"]["roots"]),
            }
        )
    if not training or not validation or not policies:
        raise ValueError("frozen own-replay splits must all be nonempty")
    features = compile_epoch_features(
        (*training, *validation, *policies), encoder, guard=guard, compact=True
    )
    return rules, encoder, tuple(training), tuple(validation), tuple(policies), features, receipts


def value_metrics(model, rows, encoder, rules, features, guard):
    nll, correct, count = 0.0, 0, 0
    model.eval()
    with torch.inference_mode():
        for first in range(0, len(rows), 256):
            guard()
            batch = rows[first : first + 256]
            tensors = _tensor_batch(model, encoder, rules, batch, "cpu", features)
            _, values = model.masked_policy_value(tensors[0], tensors[1])
            labels = tensors[8].argmax(1)
            nll += float(F.cross_entropy(values, labels, reduction="sum"))
            correct += int((values.argmax(1) == labels).sum())
            count += len(batch)
    model.train()
    return {
        "known_rows": count,
        "row_nll": nll / count,
        "row_accuracy": correct / count,
        "scope": (
            "TRAIN-only internal opening-alias split; "
            "not independent strength or optimal-play labels"
        ),
    }


def policy_kl(model, probes, encoder, rules, features, guard):
    guard()
    tensors = _tensor_batch(model, encoder, rules, probes, "cpu", features)
    with torch.inference_mode():
        logits, _ = model.masked_policy_value(tensors[0], tensors[1])
        logs = F.log_softmax(logits.masked_fill(~tensors[2], -torch.inf), 1)
        old = tensors[3]
        safe = torch.where(old > 0, logs, 0.0)
        return float((old * (old.clamp_min(1e-30).log() - safe)).sum(1).mean())


def write_checkpoint(path, payload, model):
    if path.exists():
        raise FileExistsError(path)
    temp = path.parent / ("." + path.name + ".tmp")
    temp.mkdir(parents=True, exist_ok=False)
    torch.save(payload, temp / "training.pt")
    save_weights(
        temp / "model.safetensors",
        model,
        provenance={
            "source_commit": payload["contract"]["source_commit"],
            "helper_sha256": payload["contract"]["helper_sha256"],
            "own_replay_updates": payload["accepted"],
            "teacher_labels_generated": False,
            "scope": "unpromoted own-experience candidate",
        },
    )
    manifest = {
        "schema": SCHEMA,
        "contract": payload["contract"],
        "accepted": payload["accepted"],
        "attempted": payload["attempted"],
        "artifacts": {f: sha256(temp / f) for f in ("training.pt", "model.safetensors")},
        "scope": "full optimizer/RNG/replay-index training resume; offline, no actor cursor",
    }
    (temp / "checkpoint.json").write_bytes(canonical(manifest) + b"\n")
    for p in temp.iterdir():
        with p.open("rb") as stream:
            os.fsync(stream.fileno())
    temp.rename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--journal", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--arm", choices=("value-only", "joint"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    clean_source(args.source_commit)
    if (
        args.steps <= 0
        or args.seed < 0
        or not math.isfinite(args.deadline_epoch)
        or args.deadline_epoch <= time.time()
    ):
        raise ValueError("explicit finite future deadline and positive step budget required")
    if args.stop_at is not None and not 0 < args.stop_at <= args.steps:
        raise ValueError("invalid declared process boundary")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % 2**32)
    random.seed(args.seed)
    budget = CgroupMemoryBudget(15 * 1024**3)

    def guard():
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("original CPU own-replay deadline exhausted")
        budget.check()
        if shutil.disk_usage(args.output.parent).free < 256 * 1024**2:
            raise RuntimeError("CPU own-replay disk floor violated")

    contract = {
        "source_commit": args.source_commit,
        "helper_sha256": sha256(Path(__file__)),
        "torch_version": torch.__version__,
        "arm": args.arm,
        "seed": args.seed,
        "max_steps": args.steps,
        "batch_size": 128,
        "learning_rate": 0.0005,
        "value_anchor_weight": 0.02,
        "policy_anchor_weight": 0.03,
        "joint_value_weight": 0.2,
        "policy_kl_stop": 0.02,
        "original_deadline_epoch": args.deadline_epoch,
        "initial_weights_sha256": sha256(args.weights),
        "protocol_sha256": sha256(args.protocol),
        "journal_sha256": [sha256(p) for p in args.journal],
        "native_schema": SCHEMA,
        "new_teacher_labels": False,
        "device": "cpu",
    }
    protocol = json.loads(args.protocol.read_text())
    allowed_journals = {row["sha256"] for row in protocol["journals"]}
    if (
        contract["initial_weights_sha256"] != protocol["initial_weights_sha256"]
        or args.source_commit != protocol["source_commit"]
        or not set(contract["journal_sha256"]) <= allowed_journals
        or len(contract["journal_sha256"]) != len(set(contract["journal_sha256"]))
        or args.steps not in (8, protocol["updates"])
    ):
        raise ValueError("weights/source/replay or step budget outside frozen protocol")
    if args.steps == protocol["updates"]:
        expected_source = protocol["pairing"].get(str(args.seed))
        expected = [
            r["sha256"] for r in protocol["journals"] if r["source_seed"] == expected_source
        ]
        if contract["journal_sha256"] != expected or not expected:
            raise ValueError("frozen independent training-seed/replay pairing differs")
    if args.resume is None:
        args.output.mkdir(exist_ok=False)
        (args.output / "checkpoints").mkdir()
        (args.output / "contract.json").write_bytes(canonical(contract) + b"\n")
    elif json.loads((args.output / "contract.json").read_text()) != contract:
        raise ValueError("full training resume contract changed")
    guard()
    rules, encoder, train, validation, policies, features, receipts = prepare(args.journal, guard)
    model = load_weights(args.weights).train()
    for name, p in model.named_parameters():
        p.requires_grad_(
            name.startswith(("value_", "invariant_value_", "global_value_"))
            if args.arm == "value-only"
            else not name.startswith("material_value_linear.")
        )
    frozen_parameters = {
        n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.0005,
        weight_decay=0.0001,
        foreach=False,
    )
    sampler = GameBalancedSampler(train, seed=args.seed ^ 0xBA71)
    policy_rng = random.Random(args.seed ^ 0x831A)
    probes = policies[:: max(1, len(policies) // 512)][:512]
    history, accepted, attempted = [], 0, 0
    before = value_metrics(model, validation, encoder, rules, features, guard)

    def payload():
        return {
            "schema": SCHEMA,
            "contract": contract,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "accepted": accepted,
            "attempted": attempted,
            "history": history,
            "before_validation": before,
            "dataset_receipts": receipts,
            "replay_phase": "closed-minibatch; dataset rebuilt from immutable journal hashes",
            "trainable_names": [n for n, p in model.named_parameters() if p.requires_grad],
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "value_sampler_rng": sampler.rng.getstate(),
            "policy_sampler_rng": policy_rng.getstate(),
        }

    if args.resume is None:
        write_checkpoint(args.output / "checkpoints/step-00000000", payload(), model)
    else:
        manifest = json.loads((args.resume / "checkpoint.json").read_text())
        if manifest["schema"] != SCHEMA or manifest["contract"] != contract:
            raise ValueError("native training schema or contract differs")
        if set(manifest["artifacts"]) != {"training.pt", "model.safetensors"}:
            raise ValueError("native artifact inventory differs")
        for name, digest in manifest["artifacts"].items():
            if sha256(args.resume / name) != digest:
                raise ValueError("native artifact changed")
        saved = torch.load(args.resume / "training.pt", map_location="cpu", weights_only=False)
        if (
            saved["schema"] != SCHEMA
            or saved["contract"] != contract
            or saved["dataset_receipts"] != receipts
        ):
            raise ValueError("native data/source/runtime binding differs")
        if saved["trainable_names"] != payload()["trainable_names"]:
            raise ValueError("trainable mask changed")
        if not (
            0 <= saved["accepted"] <= saved["attempted"] <= args.steps
            and len(saved["history"]) == saved["accepted"]
            and manifest["accepted"] == saved["accepted"]
            and manifest["attempted"] == saved["attempted"]
        ):
            raise ValueError("native optimizer/update counters disagree")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        accepted, attempted, history, before = (
            saved["accepted"],
            saved["attempted"],
            saved["history"],
            saved["before_validation"],
        )
        sampler.rng.setstate(saved["value_sampler_rng"])
        policy_rng.setstate(saved["policy_sampler_rng"])
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
    stop = args.stop_at or args.steps
    reason = "fixed-step-boundary"
    while accepted < stop:
        guard()
        # Full rollback includes optimizer and every sampling/global RNG state.
        old = copy.deepcopy(payload())
        value_rows = sampler.sample(128)
        vt = _tensor_batch(model, encoder, rules, value_rows, "cpu", features)
        _, values = model.masked_policy_value(vt[0], vt[1])
        value_loss = -(vt[8] * F.log_softmax(values, 1)).sum(1).mean()
        value_anchor = (
            (vt[9] * (vt[9].clamp_min(1e-30).log() - F.log_softmax(values, 1))).sum(1).mean()
        )
        loss = 0.2 * value_loss + 0.02 * value_anchor
        if args.arm == "joint":
            selected = policy_rng.choices(policies, k=128)
            pt = _tensor_batch(model, encoder, rules, selected, "cpu", features)
            logits, _ = model.masked_policy_value(pt[0], pt[1])
            logs = F.log_softmax(logits.masked_fill(~pt[2], -torch.inf), 1)
            safe_logs = torch.where(pt[2], logs, 0.0)
            targets = torch.zeros_like(logits)
            for i, row in enumerate(selected):
                targets[i, : len(row.search_policy)] = torch.tensor(row.search_policy)
            policy_loss = -(targets * safe_logs).sum(1).mean()
            anchor = (pt[4] * (pt[4].clamp_min(1e-30).log() - safe_logs)).sum(1).mean()
            loss = policy_loss + 0.2 * value_loss + 0.03 * anchor + 0.02 * value_anchor
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0
        )
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError("nonfinite own-replay loss/gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("nonfinite own-replay parameter")
        attempted += 1
        kl = policy_kl(model, probes, encoder, rules, features, guard)
        if kl > 0.02:
            model.load_state_dict(old["model"], strict=True)
            optimizer.load_state_dict(old["optimizer"])
            sampler.rng.setstate(old["value_sampler_rng"])
            policy_rng.setstate(old["policy_sampler_rng"])
            torch.set_rng_state(old["torch_rng"])
            np.random.set_state(old["numpy_rng"])
            random.setstate(old["python_rng"])
            reason = "registered-policy-KL-stop-rejected-step-preserved"
            break
        accepted += 1
        history.append(
            {
                "accepted": accepted,
                "attempted": attempted,
                "loss": float(loss.detach()),
                "value_ce": float(value_loss.detach()),
                "policy_kl": kl,
                "preclip_gradient_norm": float(norm),
            }
        )
        if accepted % 20 == 0 or accepted == stop:
            write_checkpoint(args.output / f"checkpoints/step-{accepted:08d}", payload(), model)
            print(json.dumps(history[-1]), flush=True)
    after = value_metrics(model, validation, encoder, rules, features, guard)
    frozen_unchanged = all(
        torch.equal(
            p.detach().contiguous().reshape(-1).view(torch.uint8),
            frozen_parameters[n].contiguous().reshape(-1).view(torch.uint8),
        )
        for n, p in model.named_parameters()
        if n in frozen_parameters
    )
    if not frozen_unchanged:
        raise RuntimeError("frozen policy/trunk/material parameter storage changed")
    result = {
        "schema": "cpu-own-replay-fit-result-v1",
        "status": "completed-fit-not-strength",
        "reason": reason,
        "accepted_updates": accepted,
        "attempted_updates": attempted,
        "before_validation": before,
        "after_validation": after,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "search_rows": len(policies),
        "contract": contract,
        "dataset_receipts": receipts,
        "memory_snapshot": budget.check(),
        "finished_epoch": time.time(),
        "full_online_actor_resume_claimed": False,
        "teacher_labels_generated": False,
        "frozen_parameter_storage_identical": frozen_unchanged,
        "strength_success_claimed": False,
    }
    # If a rejected step ended the run, preserve the final restored training state.
    final = args.output / f"final-attempt-{attempted:08d}"
    write_checkpoint(final, payload(), model)
    result["final_checkpoint"] = str(final)
    (args.output / f"result-attempt-{attempted:08d}.json").write_bytes(canonical(result) + b"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
