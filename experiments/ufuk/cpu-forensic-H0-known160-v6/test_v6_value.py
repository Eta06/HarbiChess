"""Actual contract metadata classification + fixture-only complete value constructor wiring."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import value
from teacher_admission import read
from value import MixedValue, classify

BASE = Path("/workspace/work/harbichess/continuation-20261007")


def protocol():
    q = read(
        {
            "path": str(BASE / "variant-known160-forensic-v5-h0/actual-dryrun-protocol.json"),
            "sha256": "b724e7389745b516b8f9b847703fe63ef1c9397a347a69e241f6fd6134ee33b7",
        }
    )
    q["runtime_revision"] = "forensic-h0-known160-runtime-v6"
    return q


@pytest.mark.parametrize("seed", [20262905, 20262906])
@pytest.mark.parametrize("role", ["learned", "parent"])
def test_actual_contract_classification_and_mocked_constructor(seed, role, monkeypatch):
    q = protocol()
    info = q["children" if role == "learned" else "parents"][str(seed)]
    c = read(info["contract"])
    packet = dict(schema="own-kingbucket-nnue16-model-v1", contract=c, model={"fixture": role})
    kind = classify(packet, q, seed, role)
    assert ("own64" if role == "learned" else "literalzero-H0") in kind
    typed = SimpleNamespace(
        SCHEMA="human-prior-own-nnue16-native-cpu-forensic-v4",
        bits_equal=lambda a, b: a == b,
        validate_weights=lambda x: None,
    )
    prior = SimpleNamespace(ClassicalValue=lambda: "prior")
    ev = SimpleNamespace(
        AuthoritativePrior=lambda *a: "authoritative",
        Evaluator=lambda *a, **kw: SimpleNamespace(nonterminal=lambda b: 0.0),
    )
    monkeypatch.setattr(value, "import_runtime", lambda p, s: (None, typed, ev, "compiled-fixture"))

    def admitted(p, s, native):
        assert native is typed and s == seed
        return {"fixture": "learned"}, {"fixture": "parent"}, {"full_native_checks": 8}

    monkeypatch.setattr(value, "child", admitted)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: packet)
    monkeypatch.setattr(value, "module", lambda *a: prior)
    endpoint = MixedValue(q["models"][str(seed)][role]["path"], q)
    assert endpoint.kind == kind
    bad = dict(packet, contract=dict(c, updates=256))
    with pytest.raises(ValueError, match="contract/source/data"):
        classify(bad, q, seed, role)


def test_legacy_v5_and_teacher256_cannot_be_reinterpreted():
    q = protocol()
    c = read(q["parents"]["20262905"]["contract"])
    packet = dict(schema="own-kingbucket-nnue16-model-v1", contract=c, model={})
    q.pop("runtime_revision")
    with pytest.raises(ValueError, match="new v6"):
        classify(packet, q, 20262905, "parent")
