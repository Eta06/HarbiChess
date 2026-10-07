"""Prospective ONE ancestry-conditional campaign:8 lowerbounds,joint nominal .00625."""

import numpy as np

ALPHA_FAMILY = 0.00625
COMPARISONS = 8
TAIL = ALPHA_FAMILY / COMPARISONS
REPLICATES = 50000


def paired_arrays(games, root_ids, adverse=False, baseline=False):
    expected = {(r, c) for r in root_ids for c in ["white", "black"]}
    seen = {}
    for game in games:
        key = (game["root_id"], game["candidate_color"])
        if key in seen or key not in expected:
            raise ValueError("exact paired root/color set")
        score = game["score"]
        if score not in [0.0, 0.5, 1.0]:
            raise ValueError("candidate POV outcome score")
        capped = game["termination"] == "max_plies"
        seen[key] = 1.0 if adverse and capped and baseline else 0.0 if adverse and capped else score
    if set(seen) != expected or len(root_ids) != 48 or len(set(root_ids)) != 48:
        raise ValueError("48 frozen roots each bothcolors")
    return np.array([(seen[r, "white"] + seen[r, "black"]) / 2 for r in root_ids], dtype=np.float64)


def bootstrap_metrics(metrics, seed):
    data = np.asarray(metrics, dtype=np.float64)
    if data.shape != (4, 48) or not np.isfinite(data).all():
        raise ValueError("four metrics48rootblocks")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 48, size=(REPLICATES, 48))
    means = data[:, draws].mean(axis=2)
    return [
        dict(
            point=float(x.mean()),
            lower=float(np.quantile(y, TAIL, method="linear")),
            upper=float(np.quantile(y, 1 - TAIL, method="linear")),
        )
        for x, y in zip(data, means, strict=True)
    ]


def evaluate_seed(arms, root_ids, seed):
    keys = {"learned-e8", "learned-parent", "learned-SF512", "e8-SF512", "parent-SF512"}
    if set(arms) != keys:
        raise ValueError("ALL FIVE arm games, no historical baseline reuse")
    raw = []
    for adverse in [False, True]:
        score = {
            k: paired_arrays(
                v, root_ids, adverse=adverse, baseline=k in ["e8-SF512", "parent-SF512"]
            )
            for k, v in arms.items()
        }
        intervals = bootstrap_metrics(
            [
                score["learned-e8"],
                score["learned-SF512"] - score["e8-SF512"],
                score["learned-parent"],
                score["learned-SF512"] - score["parent-SF512"],
            ],
            seed,
        )
        gates = dict(
            E8_direct_gt_060=intervals[0]["point"] > 0.60,
            E8_direct_lower_gt_050=intervals[0]["lower"] > 0.50,
            SF_gain_E8_gt_010=intervals[1]["point"] > 0.10,
            SF_gain_E8_lower_gt_0=intervals[1]["lower"] > 0,
            parent_direct_gt_060=intervals[2]["point"] > 0.60,
            parent_direct_lower_gt_050=intervals[2]["lower"] > 0.50,
            SF_gain_parent_gt_0=intervals[3]["point"] > 0,
            SF_gain_parent_lower_gt_0=intervals[3]["lower"] > 0,
            final_SF_ge_025=float(score["learned-SF512"].mean()) >= 0.25,
            caps_le_005=all(
                sum(g["termination"] == "max_plies" for g in games) / 96 <= 0.05
                for games in arms.values()
            ),
        )
        raw.append(
            dict(adverse_caps=adverse, intervals=intervals, gates=gates, passed=all(gates.values()))
        )
    return dict(
        seed=seed,
        passed=all(x["passed"] for x in raw),
        analyses=raw,
        root_blocks=48,
        games=480,
        confidence_two_sided=0.9984375,
        lower_tail=TAIL,
        bootstrap_replicates=REPLICATES,
        uncertainty="nominal empirical source-root block bootstrap; coverage not exact",
        trained_strength_claim=False,
        simultaneous_scope=("8 adverse-cap primary lowerbounds; raw intervals descriptive, "
                            "coupled monotone same resamples"),
    )
