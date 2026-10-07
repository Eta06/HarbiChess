"""Execute actual pre-Learner origin guards without importing training modules."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def validate(mutation=None):
    source = Path(__file__).parent / "source" / "train.py"
    tree = ast.parse(source.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "imported" for t in n.targets))
    stop = next(i for i in range(start, len(main.body)) if isinstance(main.body[i], ast.If)
                and any(isinstance(x, ast.Constant) and x.value ==
                        "imported training modules must resolve to the exact pinned source tree"
                        for x in ast.walk(main.body[i])))
    refs = {str(source.with_name(n).resolve()): "correct" for n in
            ("model.py", "native.py", "ranking.py", "train.py")}
    contract_path = source.with_name("contract.py").resolve()
    helpers = {str(contract_path): "correct"}
    imported_contract = contract_path
    if mutation == "missing":
        helpers.clear()
    elif mutation == "sha":
        helpers[str(contract_path)] = "wrong"
    elif mutation == "origin":
        imported_contract = contract_path.parent.parent / "contract.py"
    env = {"Path": Path, "__file__": str(source), "sha": lambda p: "correct",
           "contract": {"source_sha256": refs, "execution_helpers_sha256": helpers},
           "contract_module": SimpleNamespace(__file__=str(imported_contract))}
    for name in ("model", "native", "ranking"):
        env[name] = SimpleNamespace(__file__=str(source.with_name(name + ".py")))
    exec(compile(ast.fix_missing_locations(ast.Module(body=main.body[start:stop + 1],
                                                       type_ignores=[])), str(source), "exec"), env)


def test_exact_four_sources_plus_contract_helper_pass():
    validate()


@pytest.mark.parametrize("mutation", ["missing", "sha", "origin"])
def test_missing_changed_contract_helper_rejected(mutation):
    with pytest.raises(ValueError):
        validate(mutation)
