"""Pure-Python terminal/own-search scalar Adam, no E8/teacher labels or strength claim."""

import math
import random

from journal_v3 import digest, read, replay, tuple_tree
from value import FEATURE_SCALES, PRIOR, SCALE, features, model_dict

SCHEMA = "classical-own-offline-native-v3"


def prepare(path, config, protected, min_games=16, min_rows=1024):
    import chess

    state = read(path)
    packets = replay(state, config)
    known = {index: (positions, result) for _, index, positions, result in packets}
    unique = {}
    excluded = 0
    duplicate = 0
    depth = {}
    evals = {}
    for game in state["games"] + ([state["active"]] if state["active"] else []):
        for row in game["moves"]:
            depth[str(row["depth"])] = depth.get(str(row["depth"]), 0) + 1
            evals[str(row["evaluations"])] = evals.get(str(row["evaluations"]), 0) + 1
    for i, game in enumerate(state["games"]):
        root = config["roots"][game["root_index"]]
        board = chess.Board(root["root_fen"])
        touched = {" ".join(board.fen().split()[:4])}
        for u in root["prefix"] + [row["action"] for row in game["moves"]]:
            board.push_uci(u)
            touched.add(" ".join(board.fen().split()[:4]))
        if touched & set(protected):
            excluded += 1
            continue
        if i not in known:
            continue
        key = digest(
            dict(
                root_fen=root["root_fen"],
                prefix=root["prefix"],
                moves=[r["action"] for r in game["moves"]],
            )
        )
        if key in unique:
            duplicate += 1
            continue
        positions, result = known[i]
        rows = []
        for (board, mover, stored_prior), row in zip(positions, game["moves"], strict=True):
            phi = features(board)
            prior = sum(w * x for w, x in zip(PRIOR, phi, strict=True)) / SCALE
            if math.tanh(prior) != stored_prior:
                raise ValueError(
                    "stored human prior differs from exact immutable feature convention"
                )
            z = (
                0.0
                if result == "1/2-1/2"
                else 1.0
                if (result == "1-0") == (mover == "white")
                else -1.0
            )
            q = max(-1.0, min(1.0, row["search_value"]))
            rows.append(
                (tuple(x / s for x, s in zip(phi, FEATURE_SCALES, strict=True)), prior, z, q)
            )
        unique[key] = rows
    train = {k: v for k, v in unique.items() if int(k[-2:], 16) % 5 != 0}
    val = {k: v for k, v in unique.items() if int(k[-2:], 16) % 5 == 0}
    if len(unique) < min_games or sum(map(len, unique.values())) < min_rows or not train or not val:
        raise ValueError("distinct complete trajectory/known row/nonempty split admission failed")
    receipt = dict(
        schema="classical-own-data-handoff-v3",
        actions=state["actions"],
        known_games=len(packets),
        eligible_games=len(unique),
        eligible_rows=sum(map(len, unique.values())),
        training_games=len(train),
        training_rows=sum(map(len, train.values())),
        validation_games=len(val),
        validation_rows=sum(map(len, val.values())),
        protected_games_excluded=excluded,
        duplicate_aliases=duplicate,
        excluded_UNKNOWN_or_tail_rows=state["actions"] - sum(len(pos) for _, _, pos, _ in packets),
        completed_depth_histogram=depth,
        evaluations_histogram=evals,
        split_scope="realized trajectory disjoint, not source/root/position independent",
    )
    return train, val, receipt, digest(dict(groups=unique, receipt=receipt))


class Learner:
    """Fresh Adam unless an exact full explicit offline native is restored."""

    def __init__(self, seed, contract, state=None):
        self.contract = contract
        self.sampler = random.Random(seed)
        self.theta = [0.0] * 18
        self.m = [0.0] * 18
        self.v = [0.0] * 18
        self.step = 0
        random.seed(seed ^ 0xCA1B)
        if state is not None:
            if set(state) != {
                "schema",
                "contract",
                "theta",
                "m",
                "v",
                "step",
                "candidate",
                "sampler_rng",
                "global_rng",
            }:
                raise ValueError("complete exact offline native keyset required")
            if state["schema"] != SCHEMA or state["contract"] != contract:
                raise ValueError("offline native contract differs")
            if state["candidate"] != model_dict(state["theta"]):
                raise ValueError("candidate convention differs")
            for key in ("theta", "m", "v"):
                values = state[key]
                if len(values) != 18 or not all(map(math.isfinite, values)):
                    raise ValueError("invalid Adam/parameter storage")
                setattr(self, key, list(values))
            if any(x < 0 for x in self.v):
                raise ValueError("negative Adam second moments")
            self.step = state["step"]
            if (
                not isinstance(self.step, int)
                or isinstance(self.step, bool)
                or not 0 <= self.step <= contract["updates"]
            ):
                raise ValueError("native counter differs")
            self.sampler.setstate(tuple_tree(state["sampler_rng"]))
            random.setstate(tuple_tree(state["global_rng"]))

    def native(self):
        return dict(
            schema=SCHEMA,
            contract=self.contract,
            theta=self.theta,
            m=self.m,
            v=self.v,
            step=self.step,
            candidate=model_dict(self.theta),
            sampler_rng=self.sampler.getstate(),
            global_rng=random.getstate(),
        )

    def advance(self, groups, target, guard=lambda: None):
        if not self.step <= target <= self.contract["updates"]:
            raise ValueError("backward/excess update")
        groups = [groups[k] for k in sorted(groups)]
        while self.step < target:
            guard()
            grad = [0.0] * 18
            for _ in range(256):
                rows = groups[self.sampler.randrange(len(groups))]
                phi, prior, z, q = rows[self.sampler.randrange(len(rows))]
                score = math.tanh(prior + sum(t * x for t, x in zip(self.theta, phi, strict=True)))
                factor = 2 * (0.75 * (score - z) + 0.25 * (score - q)) * (1 - score * score) / 256
                for j in range(18):
                    grad[j] += factor * phi[j]
            for j in range(18):
                grad[j] += 0.02 * self.theta[j] / 18
            norm = math.sqrt(sum(g * g for g in grad))
            scale = min(1.0, 5 / max(norm, 1e-30))
            self.step += 1
            for j in range(18):
                g = grad[j] * scale
                self.m[j] = 0.9 * self.m[j] + 0.1 * g
                self.v[j] = 0.999 * self.v[j] + 0.001 * g * g
                self.theta[j] -= (
                    0.01
                    * (self.m[j] / (1 - 0.9**self.step))
                    / (math.sqrt(self.v[j] / (1 - 0.999**self.step)) + 1e-8)
                )
            if not all(map(math.isfinite, self.theta)):
                raise ValueError("nonfinite update")
        return self.native()
