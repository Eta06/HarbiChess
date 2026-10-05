"""Own-outcome additive residual control; frozen e8 prior and policy, no value reinitialization.

Training-only native v2 includes Adam, RNG and immutable dataset bindings.
This is not an online actor resume, teacher distillation, or model promotion.
"""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from features import prepare, sha

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights
from harbichess.training.cgroup_budget import CgroupMemoryBudget
from harbichess.training.torch_search_acting_learner import tensor_bits_equal
from harbichess.training.torch_search_acting_run import clean_source

SCHEMA = "cpu-own-outcome-residual-training-native-v3"
SCALE = 256.0


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def rebase(old):
    if old.architecture != "pairwise" or old.specification.get("value_sparse") is not None:
        raise ValueError("requires fixed original non-sparse e8 for additive residual")
    model = TorchChessNetwork.from_specification(
        {
            **old.specification,
            "value_sparse": {
                "schema": 2, "channels": 32, "hidden": 32, "composition": "additive-v1",
            },
        }
    )
    missing = model.load_state_dict(old.state_dict(), strict=False)
    expected = {n for n in model.state_dict() if n.startswith("value_sparse_head.")}
    if missing.unexpected_keys or set(missing.missing_keys) != expected:
        raise ValueError("unexpected architecture transfer")
    model.requires_grad_(False)
    head = model.value_sparse_head
    head.feature.requires_grad_(True)
    head.metadata.requires_grad_(True)
    with torch.no_grad():
        head.feature.weight.zero_()
        head.feature.bias.zero_()
        head.feature.bias[:3] = 0.5
        head.metadata.weight.zero_()
        head.hidden.weight.copy_(torch.eye(32))
        head.hidden.bias.zero_()
        head.output.weight.zero_()
        head.output.weight[:3, :3].copy_(torch.eye(3) * SCALE)
        head.output.bias.fill_(-SCALE / 2)
    return model


def critic(model, x, base_wdl):
    head = model.value_sparse_head
    features = (head.feature(x[:, :832]) + head.metadata(x[:, 832:])).clamp(0, 1)
    delta = head.output(head.hidden(features).clamp(0, 1))
    return base_wdl.detach().clamp_min(1e-30).log() + delta


def penalty(model):
    head = model.value_sparse_head
    pieces = (head.feature.weight[:3, :768] * SCALE).reshape(3, 64, 12)
    mean = pieces.mean(dim=1, keepdim=True)
    spatial = ((pieces - mean) ** 2).sum()
    material = (mean**2).sum()
    ep = ((head.feature.weight[:3, 768:] * SCALE) ** 2).sum()
    metadata = ((head.metadata.weight[:3, 1:] * SCALE) ** 2).sum()
    return 0.05 * (spatial + ep) + 0.001 * material + 0.01 * metadata


def verify_mask(model):
    head = model.value_sparse_head
    if any(
        torch.count_nonzero(t)
        for t in (
            head.feature.weight[3:],
            head.feature.bias[3:],
            head.metadata.weight[3:],
            head.metadata.weight[:, 0],
        )
    ):
        raise ValueError("fixed inactive coordinates changed")


def metrics(model, x, y, base_wdl, groups):
    with torch.inference_mode():
        logits = critic(model, x, base_wdl)
        losses = F.cross_entropy(logits, y, reduction="none")
        accuracy = logits.argmax(1) == y
        return {
            "complete_games": len(groups),
            "rows": sum(len(g) for g in groups),
            "game_equal_nll": float(torch.stack([losses[list(g)].mean() for g in groups]).mean()),
            "game_equal_accuracy": float(
                torch.stack([accuracy[list(g)].float().mean() for g in groups]).mean()
            ),
            "scope": (
                "source-disjoint TRAIN internal realized own-policy outcomes; "
                "not independent strength"
            ),
        }


def checkpoint(directory, payload, model):
    directory.parent.mkdir(exist_ok=True)
    temp = directory.parent / ("." + directory.name + ".tmp")
    if directory.exists() or temp.exists():
        raise FileExistsError(directory)
    temp.mkdir()
    torch.save(payload, temp / "training.pt")
    save_weights(
        temp / "model.safetensors",
        model,
        provenance={
            "source_commit": payload["contract"]["source_commit"],
            "helper_sha256": payload["contract"]["helper_sha256"],
            "value_rebase": (
                "sparse-value-v2/additive-v1 zero residual plus inherited e8 logits; fresh Adam"
            ),
            "own_outcome_updates": payload["accepted"],
            "new_teacher_labels": False,
            "scope": "unpromoted own-outcome candidate",
        },
    )
    manifest = {
        "schema": SCHEMA,
        "contract": payload["contract"],
        "accepted": payload["accepted"],
        "trainable_names": payload["trainable_names"],
        "artifacts": {f: sha(temp / f) for f in ("training.pt", "model.safetensors")},
        "scope": "full offline optimizer/RNG/immutable replay-index resume; no actor state",
    }
    (temp / "checkpoint.json").write_bytes(canonical(manifest) + b"\n")
    for file in temp.iterdir():
        with file.open("rb") as stream:
            os.fsync(stream.fileno())
    temp.rename(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "weights", "protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--journal", type=Path, action="append", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    clean_source(args.source_commit)
    if args.audit_only and args.resume is None:
        raise ValueError("audit-only requires a full native checkpoint")
    if (
        args.seed < 0
        or args.steps <= 0
        or not math.isfinite(args.deadline_epoch)
        or args.deadline_epoch <= time.time()
        or (args.stop_at is not None and not 0 < args.stop_at <= args.steps)
    ):
        raise ValueError("positive steps and observed finite original clock required")
    if torch.cuda.is_available():
        raise ValueError("this fixed study is CPU only")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % 2**32)
    random.seed(args.seed)
    budget = CgroupMemoryBudget(15 * 1024**3)

    def guard():
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("original own-outcome clock exhausted")
        budget.check()
        if shutil.disk_usage(args.output.parent).free < 256 * 1024**2:
            raise RuntimeError("registered disk floor violated")

    protocol = json.loads(args.protocol.read_text())
    journals = [sha(p) for p in args.journal]
    if (
        args.source_commit != protocol["source_commit"]
        or sha(args.weights) != protocol["initial_e8_sha256"]
        or args.steps not in (8, protocol["steps"])
        or len(journals) != len(set(journals))
        or not set(journals) <= {r["sha256"] for r in protocol["journals"]}
    ):
        raise ValueError("source/model/replay/update budget outside frozen protocol")
    if args.steps == protocol["steps"]:
        source_seed = protocol["pairing"].get(str(args.seed))
        expected = [r["sha256"] for r in protocol["journals"] if r["source_seed"] == source_seed]
        if not expected or journals != expected:
            raise ValueError("fixed train-seed/own-data pairing changed")
    contract = {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "helper_sha256": sha(Path(__file__)),
        "feature_helper_sha256": sha(Path(__file__).with_name("features.py")),
        "protocol_sha256": sha(args.protocol),
        "initial_e8_sha256": sha(args.weights),
        "journal_sha256": journals,
        "seed": args.seed,
        "max_steps": args.steps,
        "learning_rate": 0.00002,
        "weight_decay": 0.0,
        "batch_size": 256,
        "torch_version": str(torch.__version__),
        "device": "cpu",
        "original_deadline_epoch": args.deadline_epoch,
        "new_teacher_labels": False,
        "new_selfplay_moves_generated": 0,
        "anchor_kl_beta": protocol["anchor_kl_beta"],
        "value_composition": "additive-v1",
    }
    if args.resume is None:
        args.output.mkdir(exist_ok=False)
        (args.output / "contract.json").write_bytes(canonical(contract) + b"\n")
    elif json.loads((args.output / "contract.json").read_text()) != contract:
        raise ValueError("full training resume contract differs")
    x, y, base_wdl, train, validation, receipts, dataset_sha = prepare(args.journal, guard)
    split_sha = hashlib.sha256(canonical({"train": train, "validation": validation})).hexdigest()
    model = load_weights(args.weights).train()
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    model = rebase(model).train()
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all(n.startswith("value_sparse_head.") for n in names)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 26912
    verify_mask(model)
    optimizer = torch.optim.AdamW(
        model.value_sparse_head.parameters(), lr=0.00002, weight_decay=0.0, foreach=False
    )
    sampler = random.Random(args.seed ^ 0xCA1B)
    accepted, history = 0, []
    before = metrics(model, x, y, base_wdl, validation)

    def payload():
        numpy = np.random.get_state()
        return {
            "schema": SCHEMA,
            "contract": contract,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "accepted": accepted,
            "attempted": accepted,
            "trainable_names": names,
            "history": history,
            "before_validation": before,
            "dataset_sha256": dataset_sha,
            "split_sha256": split_sha,
            "dataset_receipts": receipts,
            "sampler_rng": sampler.getstate(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": {
                "kind": numpy[0],
                "keys": torch.tensor(numpy[1].astype(np.int64)),
                "position": numpy[2],
                "has_gauss": numpy[3],
                "cached_gauss": numpy[4],
            },
        }

    if args.resume is None:
        checkpoint(args.output / "checkpoints/step-00000000", payload(), model)
    else:
        manifest = json.loads((args.resume / "checkpoint.json").read_text())
        if (
            manifest["schema"] != SCHEMA
            or manifest["contract"] != contract
            or set(manifest["artifacts"]) != {"training.pt", "model.safetensors"}
            or any(sha(args.resume / f) != h for f, h in manifest["artifacts"].items())
        ):
            raise ValueError("native format/bindings/integrity differ")
        saved = torch.load(args.resume / "training.pt", map_location="cpu", weights_only=True)
        if (
            saved["schema"] != SCHEMA
            or saved["contract"] != contract
            or saved["dataset_sha256"] != dataset_sha
            or saved["split_sha256"] != split_sha
            or saved["dataset_receipts"] != receipts
            or saved["trainable_names"] != names
            or not 0
            <= saved["accepted"]
            == saved["attempted"]
            == manifest["accepted"]
            <= args.steps
            or len(saved["history"]) != saved["accepted"]
        ):
            raise ValueError("native replay-index/mask/counters mismatch")
        portable = load_weights(args.resume / "model.safetensors").state_dict()
        if not all(tensor_bits_equal(v, portable[n]) for n, v in saved["model"].items()):
            raise ValueError("native versus portable model bits differ")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        accepted, history, before = saved["accepted"], saved["history"], saved["before_validation"]
        sampler.setstate(saved["sampler_rng"])
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        n = saved["numpy_rng"]
        np.random.set_state(
            (
                n["kind"],
                n["keys"].numpy().astype(np.uint32),
                n["position"],
                n["has_gauss"],
                n["cached_gauss"],
            )
        )
        if accepted and (
            len(optimizer.state) != len(names)
            or any(int(s["step"]) != accepted for s in optimizer.state.values())
        ):
            raise ValueError("restored Adam steps disagree with accepted counter")
    stop = accepted if args.audit_only else (args.stop_at or args.steps)
    while accepted < stop:
        guard()
        selected = [sampler.choice(g) for g in sampler.choices(train, k=256)]
        logits = critic(model, x[selected], base_wdl[selected])
        ce = F.cross_entropy(logits, y[selected])
        regularization = penalty(model)
        anchor = base_wdl[selected].detach()
        anchor_kl = (
            anchor * (anchor.clamp_min(1e-30).log() - F.log_softmax(logits, dim=1))
        ).sum(1).mean()
        loss = ce + protocol["anchor_kl_beta"] * anchor_kl + regularization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.value_sparse_head.parameters(), 5.0)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError("nonfinite own-outcome loss or gradient")
        model.value_sparse_head.metadata.weight.grad[:, 0].zero_()
        optimizer.step()
        verify_mask(model)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("nonfinite model storage")
        accepted += 1
        history.append(
            {
                "step": accepted,
                "loss": float(loss.detach()),
                "cross_entropy": float(ce.detach()),
                "regularization": float(regularization.detach()),
                "fixed_e8_kl": float(anchor_kl.detach()),
                "preclip_norm": float(norm),
            }
        )
        if accepted % 512 == 0 or accepted == stop or (args.steps == 8 and accepted == 4):
            checkpoint(args.output / f"checkpoints/step-{accepted:08d}", payload(), model)
            print(json.dumps(history[-1]), flush=True)
    if not all(tensor_bits_equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen):
        raise ValueError("frozen storage changed from explicit rebased baseline")
    if not all(
        tensor_bits_equal(p, original[n])
        for n, p in model.named_parameters()
        if n.startswith(("stem.", "blocks.", "pair_"))
    ):
        raise ValueError("original policy/trunk storage changed")
    verify_mask(model)
    result = {
        "schema": "cpu-own-outcome-residual-fit-v1",
        "status": "completed-fit-not-strength",
        "accepted_updates": accepted,
        "before_validation": before,
        "after_validation": metrics(model, x, y, base_wdl, validation),
        "contract": contract,
        "dataset_receipts": receipts,
        "dataset_sha256": dataset_sha,
        "split_sha256": split_sha,
        "final_checkpoint": str(args.output / f"checkpoints/step-{accepted:08d}"),
        "policy_trunk_storage_unchanged": True,
        "requires_grad_storage_parameters": 26912,
        "effective_trainable_coordinates": 2520,
        "inactive_coordinates_verified_zero": True,
        "frozen_inherited_e8_storage_unchanged": True,
        "zero_residual_control": (
            "exacte8 baseWDL and policy, newhead/freshAdam weights-only adaptation"
        ),
        "value_reinitialization_claimed_as_learning": False,
        "full_online_actor_resume": False,
        "new_teacher_labels": False,
        "strength_success_claimed": False,
        "memory_snapshot": budget.check(),
        "finished_epoch": time.time(),
    }
    if not args.audit_only:
        path = args.output / f"result-step-{accepted:08d}-invocation-{time.time_ns()}.json"
        with path.open("xb") as stream:
            stream.write(canonical(result) + b"\n")
    else:
        result["status"] = "PASS-fresh-full-training-native-load-no-update"
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "accepted_updates", "before_validation", "after_validation")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
