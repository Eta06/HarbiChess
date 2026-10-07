"""Real-shape parent helper binding; no model imports, search or SGD."""
import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).with_name("seal_factory_v4.py")


def inference_refs(tmp_path, mutation=None):
    tree = ast.parse(SOURCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "make_seal")
    start = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "helper_dir" for t in n.targets))
    stop = next(i for i in range(start, len(function.body))
                if isinstance(function.body[i], ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "seal"
                        for t in function.body[i].targets))
    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    helpers = {"directory": str(tmp_path)}
    for name in ("model", "native", "evaluator"):
        path = tmp_path / (name + ".py")
        path.write_text("actual-shape fixture " + name)
        helpers[name + "_sha256"] = digest(path)
    extension = tmp_path / "extension.so"
    extension.write_bytes(b"fixture")
    helpers.update(extension_path=str(extension), extension_sha256=digest(extension),
                   prior_path=str(tmp_path / "prior.py"), prior_sha256="fixture-prior")
    assert "evaluator_path" not in helpers
    if mutation == "digest":
        helpers["evaluator_sha256"] = "0" * 64
    elif mutation == "file":
        (tmp_path / "evaluator.py").write_text("tampered")
    elif mutation == "model":
        helpers["model_sha256"] = "0" * 64
    elif mutation == "native":
        helpers["native_sha256"] = "0" * 64
    elif mutation == "extension":
        helpers["extension_sha256"] = "0" * 64
    env = {"Path": Path, "registration": {"parent_helpers": helpers}, "sha": digest}
    exec(compile(ast.fix_missing_locations(ast.Module(body=function.body[start:stop],
                                                       type_ignores=[])), str(SOURCE), "exec"), env)
    return env["inference_refs"]


def test_actual_registration_shape_derives_evaluator_from_directory(tmp_path):
    refs = inference_refs(tmp_path)
    assert str(tmp_path / "evaluator.py") in refs
    assert len(refs) == 4


@pytest.mark.parametrize("mutation", ["digest", "file", "model", "native", "extension"])
def test_full_inference_source_tampering_rejected(tmp_path, mutation):
    with pytest.raises(ValueError, match="parent inference source differs"):
        inference_refs(tmp_path, mutation)


def test_existing_successful_v3_converter_and_audit_paths_retained():
    desc = importlib.util.spec_from_file_location("metadata_factory_v4_fixture", SOURCE)
    module = importlib.util.module_from_spec(desc)
    desc.loader.exec_module(module)
    original = (
        Path(__file__).resolve().parent.parent / "closed-terminal-own-v3-producer-converter-binding"
    )
    assert original / "source" == module.SOURCE
    text = SOURCE.read_text()
    assert 'expected_converter = SOURCE / "convert.py"' in text
    assert 'ref(V3_ROOT / "integration" / "audit_six.py")' in text
