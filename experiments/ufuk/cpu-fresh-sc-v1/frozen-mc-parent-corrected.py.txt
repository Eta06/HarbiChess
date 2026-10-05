"""Own-outcome additive residual control; frozen e8 prior and policy, no value reinitialization.

Training-only native v2 includes Adam, RNG and immutable dataset bindings.
This is not an online actor resume, teacher distillation, or model promotion.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F
from features import sha

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights
from harbichess.training.cgroup_budget import CgroupMemoryBudget
from harbichess.training.torch_search_acting_learner import tensor_bits_equal
from harbichess.training.torch_search_acting_run import clean_source

SCHEMA = "fresh-qsearch-additive-own-mc-native-v2"
SCALE = 256.0
MAX_ARTIFACT_BYTES = 512 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def rebase(old):
    if old.architecture != "pairwise" or old.specification.get("value_sparse") is not None:
        raise ValueError("requires fixed original non-sparse e8 for additive residual")
    model = TorchChessNetwork.from_specification(
        {
            **old.specification,
            "value_sparse": {
                "schema": 2,
                "channels": 32,
                "hidden": 32,
                "composition": "additive-v1",
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


def load_pinned(path, expected_sha, name):
    if sha(path) != expected_sha:
        raise ValueError(f"{name} helper SHA differs")
    spec = importlib.util.spec_from_file_location(f"{name}_{expected_sha}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def trajectory_key(root, game):
    return hashlib.sha256(
        canonical(
            {
                "root_fen": root["root_fen"],
                "prefix": root["prefix"],
                "moves": [row["action"] for row in game["moves"]],
            }
        )
    ).hexdigest()


def position_key(board):
    return " ".join(board.fen().split()[:4])


def touches_protected_position(root, game, protected):
    board = chess.Board(root["root_fen"])
    touched = {position_key(board)}
    for move_text in (*root["prefix"], *(row["action"] for row in game["moves"])):
        board.push_uci(move_text)
        touched.add(position_key(board))
    return bool(touched.intersection(protected))


def prepare_fresh(
    journal_path,
    expected_journal_sha,
    actor_config_path,
    expected_config_sha,
    journal_helper_path,
    journal_helper_sha,
    feature_helper_path,
    feature_helper_sha,
    minimum_rows=1024,
    minimum_games=16,
    protected_position_keys=(),
):
    if sha(journal_path) != expected_journal_sha or sha(actor_config_path) != expected_config_sha:
        raise ValueError("immutable v2 journal/config input SHA differs")
    actor_config = json.loads(actor_config_path.read_text())
    protected = frozenset(str(key) for key in protected_position_keys)
    configured_protected = actor_config.get("excluded_training_position_keys")
    book_sha = actor_config.get("exclusion_book_sha256")
    if (
        not isinstance(configured_protected, list)
        or not configured_protected
        or configured_protected != sorted(set(configured_protected))
        or sorted(protected) != configured_protected
        or not isinstance(book_sha, str)
        or len(book_sha) != 64
    ):
        raise ValueError("protocol protected keys differ from frozen actor exclusion config")
    for key in protected:
        if len(key.split()) != 4:
            raise ValueError("protected position key must be canonical FEN4")
        board = chess.Board(key + " 0 1")
        if not board.is_valid() or position_key(board) != key:
            raise ValueError("protected position key is invalid or noncanonical")
    journal_module = load_pinned(journal_helper_path, journal_helper_sha, "journal_v2")
    if journal_module.SCHEMA != "fresh-qsearch-selfplay-journal-v2":
        raise ValueError("journal helper is not explicit v2")
    x, y, anchors, groups, provenance, receipt = journal_module.convert_verified(
        journal_path,
        expected_journal_sha,
        actor_config,
        feature_helper_path,
        feature_helper_sha,
    )
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    anchors = np.asarray(anchors, dtype=np.float32)
    groups = tuple(tuple(int(index) for index in group) for group in groups)
    if (
        x.shape != (len(y), 840)
        or anchors.shape != (len(y), 3)
        or len(groups) != len(provenance)
        or len(groups) != receipt["complete_game_groups"]
        or len(y) != receipt["known_rows"]
        or receipt["schema"] != "fresh-qsearch-to-shrink840-handoff-v2"
        or receipt["anchor_model_sha256"] != actor_config["anchor_model_sha256"]
        or receipt["anchor_helper_sha256"] != actor_config["anchor_helper_sha256"]
        or receipt["anchor_target"] != actor_config["anchor_target"]
    ):
        raise ValueError("v2 data/anchor converter receipt shape differs")
    if not np.isfinite(x).all() or not np.isfinite(anchors).all():
        raise ValueError("nonfinite v2 feature or frozen base anchor")
    if not np.isin(y, (0, 1, 2)).all() or np.any(anchors < 0):
        raise ValueError("invalid own WDL labels or anchor probabilities")
    if not np.allclose(anchors.sum(axis=1), 1.0, atol=2e-6, rtol=0):
        raise ValueError("v2 base anchors do not normalize")
    if sorted(i for group in groups for i in group) != list(range(len(y))):
        raise ValueError("complete game groups do not partition known rows")

    # Split by the realized full trajectory. All roots are ordinary starts and share
    # one source_id, so source-id hashing cannot define a useful validation split.
    # This remains game-disjoint internal validation, not root/source/state independence.
    state = journal_module.read(journal_path)
    journal_module.replay(state, actor_config)
    all_games = state["games"]
    by_index = {
        int(item["game_index"]): tuple(group)
        for item, group in zip(provenance, groups, strict=True)
    }
    unique = {}
    protected_game_count = 0
    protected_known_rows = 0
    duplicate_trajectory_count = 0
    duplicate_known_rows = 0
    protected_exclusion_ledger = []
    for game_index, game in enumerate(all_games):
        root = actor_config["roots"][game["root_index"]]
        if touches_protected_position(root, game, protected):
            protected_game_count += 1
            protected_known_rows += len(by_index.get(game_index, ()))
            protected_exclusion_ledger.append(
                {
                    "game_index": game_index,
                    "trajectory_sha256": trajectory_key(root, game),
                    "known_rows_removed": len(by_index.get(game_index, ())),
                    "result_status": game["result"],
                    "termination": game["termination"],
                }
            )
            continue
        if game_index not in by_index:
            continue  # UNKNOWN/capped complete game has no outcome-labelled rows.
        trajectory = trajectory_key(root, game)
        if trajectory in unique:
            duplicate_trajectory_count += 1
            duplicate_known_rows += len(by_index[game_index])
        else:
            unique[trajectory] = {"indices": list(by_index[game_index])}

    if not unique:
        raise ValueError("no eligible complete trajectories after protected-FEN exclusion")
    # Exact duplicate trajectories add no independent experience; retain the first
    # trajectory's rows and its shared digest, independent of source/seed/game IDs.
    trajectory_ids = tuple(sorted(unique))
    selected_groups = tuple(tuple(unique[key]["indices"]) for key in trajectory_ids)

    if len(selected_groups) < minimum_games or sum(map(len, selected_groups)) < minimum_rows:
        raise ValueError(
            "fresh-data admission requires "
            f"{minimum_games} distinct complete trajectories and {minimum_rows} known rows"
        )
    # One deterministic bucket per complete trajectory; identical trajectories are
    # deduplicated before partitioning. Bucket 0 is validation.
    train_groups = tuple(i for i, key in enumerate(trajectory_ids) if int(key[-2:], 16) % 5 != 0)
    validation_groups = tuple(
        i for i, key in enumerate(trajectory_ids) if int(key[-2:], 16) % 5 == 0
    )
    if not train_groups or not validation_groups:
        raise ValueError("trajectory-hash train/validation split is empty")
    train_indices = tuple(index for gi in train_groups for index in selected_groups[gi])
    validation_indices = tuple(index for gi in validation_groups for index in selected_groups[gi])
    if set(train_indices) & set(validation_indices):
        raise ValueError("trajectory-partition leakage")
    groups = tuple(selected_groups)
    trajectory_groups = trajectory_ids
    receipts = [receipt]
    receipts.append(
        {
            "schema": "fresh-additive-trajectory-split-v1",
            "split": (
                "last 2 hex digits of sha256(full root FEN + prefix + "
                "realized played moves) mod 5; bucket 0 validation"
            ),
            "independence_scope": (
                "generated-trajectory-disjoint internal validation; "
                "not root/source-family/state-disjoint"
            ),
            "protected_position_keys_sha256": hashlib.sha256(
                canonical(sorted(protected))
            ).hexdigest(),
            "protected_position_book_sha256": book_sha,
            "protected_position_key_count": len(protected),
            "excluded_complete_games_touching_protected_positions": protected_game_count,
            "protected_exclusion_ledger_sha256": hashlib.sha256(
                canonical(protected_exclusion_ledger)
            ).hexdigest(),
            "deduplicated_identical_trajectory_aliases": duplicate_trajectory_count,
            "duplicate_known_rows_removed": duplicate_known_rows,
            "protected_known_rows_removed": protected_known_rows,
            "eligible_known_rows": sum(map(len, groups)),
            "eligible_distinct_trajectories": len(groups),
            "source_ids_are_not_split_keys": True,
            "split_hash_slice": "last-2-hex-digits",
        }
    )
    dataset_sha = hashlib.sha256(
        x.tobytes()
        + y.tobytes()
        + anchors.tobytes()
        + canonical(groups)
        + canonical(trajectory_groups)
        + canonical(receipts)
    ).hexdigest()
    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(anchors),
        groups,
        trajectory_groups,
        train_groups,
        validation_groups,
        train_indices,
        validation_indices,
        receipts,
        dataset_sha,
        actor_config,
    )


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
                "trajectory-disjoint TRAIN internal realized own-policy outcomes; "
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
    if (temp / "training.pt").stat().st_size > MAX_ARTIFACT_BYTES:
        raise RuntimeError("native training payload exceeds 512KiB artifact ceiling")
    manifest = {
        "schema": SCHEMA,
        "contract": payload["contract"],
        "accepted": payload["accepted"],
        "trainable_names": payload["trainable_names"],
        "artifacts": {"training.pt": sha(temp / "training.pt")},
        "scope": (
            "full offline optimizer/RNG/head-state resume against SHA-bound frozen e8; "
            "no actor state"
        ),
    }
    (temp / "checkpoint.json").write_bytes(canonical(manifest) + b"\n")
    for file in temp.iterdir():
        with file.open("rb") as stream:
            os.fsync(stream.fileno())
    temp.rename(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output",
        "weights",
        "protocol",
        "journal",
        "actor-config",
        "journal-helper",
        "feature-helper",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
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
        or not math.isfinite(args.deadline_epoch)
        or args.deadline_epoch <= time.time()
    ):
        raise ValueError("positive seed and observed finite original clock required")
    if torch.cuda.is_available():
        raise ValueError("fresh own-MC training protocol is CPU only")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % 2**32)
    random.seed(args.seed)
    budget = CgroupMemoryBudget(15 * 1024**3)

    def guard():
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("registered original QSEARCH training clock exhausted")
        budget.check()
        if shutil.disk_usage(args.output.parent).free < 256 * 1024**2:
            raise RuntimeError("registered 256MiB disk floor violated")

    protocol = json.loads(args.protocol.read_text())
    protocol_sha = sha(args.protocol)
    registered_deadline = protocol.get("deadline_epoch_by_seed", {}).get(str(args.seed))
    if (
        protocol.get("schema") != "fresh-qsearch-additive-own-mc-training-protocol-v2"
        or args.source_commit != protocol["source_commit"]
        or sha(Path(__file__)) != protocol["trainer_sha256"]
        or sha(args.weights) != protocol["initial_e8_sha256"]
        or args.seed not in protocol["seeds"]
        or protocol["anchor_kl_beta"] != 1.0
        or protocol["learning_rate"] != 0.00002
        or protocol["batch_size"] != 256
        or protocol["presentation_slots_per_train_row"] != 4
        or protocol["max_native_or_candidate_artifact_bytes"] != MAX_ARTIFACT_BYTES
        or protocol["protected_position_keys"] != sorted(set(protocol["protected_position_keys"]))
        or hashlib.sha256(canonical(protocol["protected_position_keys"])).hexdigest()
        != protocol["protected_position_keys_sha256"]
        or registered_deadline != args.deadline_epoch
        or args.deadline_epoch <= time.time()
    ):
        raise ValueError("source/model/objective/update contract outside frozen protocol")
    journal_sha = sha(args.journal)
    actor_config_sha = sha(args.actor_config)
    actor_config_bound = json.loads(args.actor_config.read_text())
    if protocol["protected_position_book_sha256"] != actor_config_bound.get(
        "exclusion_book_sha256"
    ) or protocol["protected_position_keys"] != actor_config_bound.get(
        "excluded_training_position_keys"
    ):
        raise ValueError("protected-position book/list differs from immutable actor config")
    pair = protocol["seed_pairing"].get(str(args.seed))
    if (
        not pair
        or pair["journal_sha256"] != journal_sha
        or pair["actor_config_sha256"] != actor_config_sha
    ):
        raise ValueError("fresh data is not the preregistered per-seed journal/config")
    data = prepare_fresh(
        args.journal,
        journal_sha,
        args.actor_config,
        actor_config_sha,
        args.journal_helper,
        protocol["journal_helper_sha256"],
        args.feature_helper,
        protocol["feature_helper_sha256"],
        protected_position_keys=protocol["protected_position_keys"],
    )
    (
        x,
        y,
        base_wdl,
        groups,
        trajectory_groups,
        train_groups,
        validation_groups,
        train,
        validation,
        receipts,
        dataset_sha,
        actor_config,
    ) = data
    if actor_config["seed"] != args.seed:
        raise ValueError("training seed differs from original frozen actor seed")
    if (
        actor_config["model_sha256"] != protocol["initial_e8_sha256"]
        or actor_config["anchor_model_sha256"] != protocol["initial_e8_sha256"]
        or actor_config["anchor_helper_sha256"] != protocol["anchor_helper_sha256"]
        or actor_config["anchor_target"] != "frozen-e8-wdl-probabilities-mover-perspective-v1"
        or actor_config["source_commit"] != args.source_commit
        or actor_config["torch_version"] != str(torch.__version__)
    ):
        raise ValueError("actor and frozen anchor must be the exact registered e8 snapshot")
    updates = min(1024, (4 * len(train)) // 256)
    if updates <= 0 or (args.stop_at is not None and not 0 < args.stop_at <= updates):
        raise ValueError("derived update budget or requested stop is invalid")
    contract = {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "trainer_sha256": sha(Path(__file__)),
        "feature_helper_sha256": protocol["feature_helper_sha256"],
        "journal_helper_sha256": protocol["journal_helper_sha256"],
        "anchor_helper_sha256": actor_config["anchor_helper_sha256"],
        "protocol_sha256": protocol_sha,
        "actor_config_sha256": actor_config_sha,
        "actor_config_sha256_inside_receipt": receipts[0]["source_config_sha256"],
        "actor_model_sha256": actor_config["model_sha256"],
        "frozen_e8_sha256": sha(args.weights),
        "journal_sha256": journal_sha,
        "dataset_sha256": dataset_sha,
        "seed": args.seed,
        "max_updates": updates,
        "sampling": "uniform-complete-game-then-uniform-position;4*N_train total slots",
        "learning_rate": 0.00002,
        "weight_decay": 0.0,
        "batch_size": 256,
        "clip_norm": 5.0,
        "torch_version": str(torch.__version__),
        "device": "cpu",
        "threads": 1,
        "original_deadline_epoch": args.deadline_epoch,
        "new_teacher_labels": False,
        "new_selfplay_moves_generated": 0,
        "anchor_kl_beta": 1.0,
        "value_composition": "log(frozen-e8-WDL)+sparse-residual-logits",
        "data_generation": "fresh-QSEARCH-v2-own-terminal-data; not old-dataset resume",
    }
    split_sha = hashlib.sha256(canonical({"train": train, "validation": validation})).hexdigest()
    if args.resume is None:
        args.output.mkdir(exist_ok=False)
        (args.output / "contract.json").write_bytes(canonical(contract) + b"\n")
    elif json.loads((args.output / "contract.json").read_text()) != contract:
        raise ValueError("full native contract/input/dataset changed")

    base_model = load_weights(args.weights).eval()
    original = {n: p.detach().clone() for n, p in base_model.named_parameters()}
    model = rebase(base_model).train()
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names or any(not n.startswith("value_sparse_head.") for n in names):
        raise ValueError("only sparse additive value parameters may train")
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != 26912:
        raise ValueError("unexpected additive residual parameterization")
    verify_mask(model)
    optimizer = torch.optim.AdamW(
        model.value_sparse_head.parameters(), lr=0.00002, weight_decay=0.0, foreach=False
    )
    sampler = random.Random(args.seed ^ 0xCA1B)
    accepted, history = 0, []
    before = metrics(model, x, y, base_wdl, [groups[i] for i in validation_groups])

    def payload():
        numpy_state = np.random.get_state()
        return {
            "schema": SCHEMA,
            "contract": contract,
            "dataset_sha256": dataset_sha,
            "split_sha256": split_sha,
            "dataset_receipts": receipts,
            "trainable_names": names,
            "head_state": model.value_sparse_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "accepted": accepted,
            "attempted": accepted,
            "before_validation": before,
            "sampler_rng": sampler.getstate(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": {
                "kind": numpy_state[0],
                "keys": torch.tensor(numpy_state[1].astype(np.int64)),
                "position": numpy_state[2],
                "has_gauss": numpy_state[3],
                "cached_gauss": numpy_state[4],
            },
        }

    if args.resume is None:
        checkpoint(args.output / "checkpoints/step-00000000", payload(), model)
    else:
        manifest = json.loads((args.resume / "checkpoint.json").read_text())
        if (
            manifest["schema"] != SCHEMA
            or manifest["contract"] != contract
            or set(manifest["artifacts"]) != {"training.pt"}
            or any(
                sha(args.resume / filename) != digest
                for filename, digest in manifest["artifacts"].items()
            )
        ):
            raise ValueError("native artifact or data/source binding differs")
        saved = torch.load(args.resume / "training.pt", map_location="cpu", weights_only=True)
        if (
            saved["schema"] != SCHEMA
            or saved["contract"] != contract
            or saved["dataset_sha256"] != dataset_sha
            or saved["split_sha256"] != split_sha
            or saved["dataset_receipts"] != receipts
            or saved["trainable_names"] != names
            or not 0 <= saved["accepted"] == saved["attempted"] == manifest["accepted"] <= updates
        ):
            raise ValueError("native data/split/mask/counter contract differs")
        model.value_sparse_head.load_state_dict(saved["head_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        accepted, history, before = saved["accepted"], [], saved["before_validation"]
        sampler.setstate(saved["sampler_rng"])
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        numpy = saved["numpy_rng"]
        np.random.set_state(
            (
                numpy["kind"],
                numpy["keys"].numpy().astype(np.uint32),
                numpy["position"],
                numpy["has_gauss"],
                numpy["cached_gauss"],
            )
        )
        if accepted and (
            len(optimizer.state) != len(names)
            or any(int(state["step"]) != accepted for state in optimizer.state.values())
        ):
            raise ValueError("restored Adam cursor differs")

    stop = accepted if args.audit_only else (args.stop_at or updates)
    while accepted < stop:
        guard()
        selected = [sampler.choice(groups[sampler.choice(train_groups)]) for _ in range(256)]
        logits = critic(model, x[selected], base_wdl[selected])
        ce = F.cross_entropy(logits, y[selected])
        regularization = penalty(model)
        anchor = base_wdl[selected].detach()
        anchor_kl = (
            (anchor * (anchor.clamp_min(1e-30).log() - F.log_softmax(logits, 1))).sum(1).mean()
        )
        loss = ce + anchor_kl + regularization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.value_sparse_head.parameters(), 5.0)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError("nonfinite fresh own-MC loss or gradient")
        model.value_sparse_head.metadata.weight.grad[:, 0].zero_()
        optimizer.step()
        verify_mask(model)
        if any(not torch.isfinite(param).all() for param in model.parameters()):
            raise RuntimeError("nonfinite model storage")
        accepted += 1
        history.append(
            {
                "step": accepted,
                "loss": float(loss.detach()),
                "own_mc_ce": float(ce.detach()),
                "fixed_e8_kl": float(anchor_kl.detach()),
                "regularization": float(regularization.detach()),
                "preclip_norm": float(norm),
            }
        )
        if accepted % 512 == 0 or accepted == stop or (stop <= 8 and accepted == 4):
            checkpoint(args.output / f"checkpoints/step-{accepted:08d}", payload(), model)
            print(json.dumps(history[-1]), flush=True)

    if not all(
        tensor_bits_equal(p, frozen[name]) for name, p in model.named_parameters() if name in frozen
    ):
        raise ValueError("frozen e8 base/policy/trunk storage changed")
    if not all(
        tensor_bits_equal(param, original[name])
        for name, param in model.named_parameters()
        if name.startswith(("stem.", "blocks.", "pair_"))
    ):
        raise ValueError("base policy/trunk changed")
    verify_mask(model)
    result = {
        "schema": "fresh-qsearch-additive-own-mc-fit-v2",
        "status": "native-proof-not-candidate" if stop < updates else "completed-fit-not-strength",
        "accepted_updates": accepted,
        "derived_update_budget": updates,
        "eligible_known_rows": len(train) + len(validation),
        "known_rows_from_converter_before_exclusions": len(y),
        "train_rows": len(train),
        "complete_games": len(groups),
        "before_validation": before,
        "after_validation": metrics(model, x, y, base_wdl, [groups[i] for i in validation_groups]),
        "contract": contract,
        "dataset_receipts": receipts,
        "dataset_sha256": dataset_sha,
        "split_sha256": split_sha,
        "policy_trunk_unchanged": True,
        "effective_trainable_coordinates": 2520,
        "frozen_e8_storage_unchanged": True,
        "new_teacher_labels": False,
        "strength_success_claimed": False,
        "memory_snapshot": budget.check(),
        "finished_epoch": time.time(),
    }
    if not args.audit_only:
        if stop == updates:
            save_weights(
                args.output / "candidate.safetensors",
                model,
                provenance={
                    "native_schema": SCHEMA,
                    "dataset_sha256": dataset_sha,
                    "updates": accepted,
                    "strength_claimed": False,
                },
            )
            if (args.output / "candidate.safetensors").stat().st_size > MAX_ARTIFACT_BYTES:
                raise RuntimeError("portable additive candidate exceeds 512KiB artifact ceiling")
        result_path = args.output / f"result-step-{accepted:08d}-invocation-{time.time_ns()}.json"
        result_path.write_bytes(canonical(result) + b"\n")
    elif stop != accepted:
        raise AssertionError("audit-only changed update cursor")
    print(
        json.dumps({k: result[k] for k in ("status", "accepted_updates", "derived_update_budget")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
