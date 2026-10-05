"""Same Q512, humanprior/E8 evaluators, only learned interior quiet ordering."""

import hashlib
import importlib.util
import sys
from pathlib import Path

BASE_SHA = "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670"
QUIET = Path("/workspace/work/harbichess/cpu-quiet-ordering-proposal")
ORDER_SHA = "b8f8815bb7fb21d83e1316236695290f0f4e55af0a1cea7c9cda86a4343036e4"
MODEL_SHA = "98c84a9b7786b4c95f6a13e07e9ed102e071ddde04a744152fe8d034dc4da5f6"
basepath = Path(__file__).with_name("search_base.py")
if hashlib.sha256(basepath.read_bytes()).hexdigest() != BASE_SHA:
    raise ValueError("original allroot search bytes differ")
for name, expected in [("ordered_search.py", ORDER_SHA), ("model.py", MODEL_SHA)]:
    if hashlib.sha256((QUIET / name).read_bytes()).hexdigest() != expected:
        raise ValueError("qualified ordering/model helper differs")
sys.path.insert(0, str(QUIET))
ordering = importlib.import_module("ordered_search")
if Path(ordering.__file__).resolve() != (QUIET / "ordered_search.py").resolve():
    raise ValueError("ordering import origin differs")

spec = importlib.util.spec_from_file_location("quiet_arena_original_search", basepath)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
QuietBase = ordering.search_type(base.BudgetSearch)


class BudgetSearch(QuietBase):
    def __init__(self, evaluator, **kwargs):
        super().__init__(
            evaluator, ordering_weights=getattr(evaluator, "ordering_weights", None), **kwargs
        )
