"""Known160 typed compiled child/actualzero; original E8 adapter unchanged."""

from pathlib import Path

from admission import child
from compiled_evaluator import Evaluator
from support import module, pinned, read
from zero_parent import admit


class MixedValue:
    def __init__(self, path, q):
        if q["schema"] != "antisymmetric-procedural-known160-protocol-v3":
            raise ValueError("typed new family arena, no old schema reinterpretation")
        matches = [
            (int(seed), role, ref)
            for seed, roles in q["models"].items()
            for role, ref in roles.items()
            if Path(ref["path"]).resolve() == Path(path).resolve()
        ]
        if not matches or len({(role, item["sha256"]) for _, role, item in matches}) != 1:
            raise ValueError("exact endpoint seed/role identity; shared E8 same-role allowed")
        if len(matches) > 1 and matches[0][1] != "e8":
            raise ValueError("own endpoint cannot stand in for two distinct seed lineages")
        seed, role, ref = matches[0]
        pinned(ref)
        if role == "e8":
            if (
                ref["sha256"] != "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
                or q["original_e8_value"]["sha256"]
                != "2abb3ac2812853c19abb001a0d38a9a1874115b8ab272cc9322bf7123f6cfd84"
            ):
                raise ValueError("unchanged pinned original E8 and original adapter")
            original = module(q["original_e8_value"], "antisymmetric_original_E8_value")
            self.value = original.MixedValue(path, read(q["original_e8_protocol"]))
            return
        if role == "learned":
            endpoint = q["children"][str(seed)]
            packet, c = child(endpoint)
            if ref != endpoint["candidate"]:
                raise ValueError("admitted native64 candidate only")
            state = packet["model"]
        elif role == "parent":
            endpoint = q["zeros"][str(seed)]
            native, c = admit(endpoint, seed)
            if ref != endpoint["candidate"]:
                raise ValueError("actual typed literalzero candidate only")
            state = native["model"]
        else:
            raise ValueError("exact learned/parent/e8 roles")
        if c["search_helper"] != q["original_search"] or c["prior_helper"] != q["prior_helper"]:
            raise ValueError("same original search and authoritativehuman18")
        self.value = Evaluator(state, compiled_refs=c["compiled_refs"], prior_ref=c["prior_helper"])

    def __call__(self, board):
        return self.value(board)
