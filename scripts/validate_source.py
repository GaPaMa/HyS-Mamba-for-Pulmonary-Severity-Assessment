"""Validate source packaging without importing PyTorch or downloading weights."""
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import re
import runpy


ROOT = Path(__file__).resolve().parents[1]


def validate():
    files = list(ROOT.rglob("*.py"))
    for path in files:
        compile(path.read_text(), str(path), "exec")
    print(f"Python syntax: {len(files)} files passed")

    manifest = json.loads((ROOT / "docs/source_manifest.json").read_text())
    for entry in manifest:
        path = ROOT / entry["path"]
        assert path.is_file(), f"Missing supplied source: {path}"
        if "packaged_sha256" in entry:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["packaged_sha256"], path
    print("Supplied source manifest: passed")

    for path in (ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("lung_severity_assesmnt"):
                target = ROOT / "src" / node.module.replace(".", "/")
                assert target.with_suffix(".py").is_file() or (target / "__init__.py").is_file(), (path, node.module)
    print("Absolute package import targets: passed")

    previous = os.environ.get("HYSMAMBA_DATA_ROOT")
    try:
        os.environ["HYSMAMBA_DATA_ROOT"] = "/tmp/hysmamba-validation-data"
        configs = {name: runpy.run_path(str(ROOT / "configs" / name)) for name in
                   ["dino_ViM_CT.py", "dino_ViM_ralo.py", "dino_ViM_ralo_512.py"]}
        for name, cfg in configs.items():
            for split in ["train", "valid", "test"]:
                data = cfg[f"{split}_dataset"]
                assert data["path"] == "/tmp/hysmamba-validation-data", (name, split)
                assert data["ann_file"].startswith(data["path"] + "/"), (name, split)
            assert cfg["model"]["projection_mode"] == "simple"
            assert cfg["model"]["fusion_mode"] == "damamba"
            assert cfg["model"]["use_coag"] is True
        assert configs["dino_ViM_CT.py"]["model"]["output_activation"] is None
        assert configs["dino_ViM_CT.py"]["model"]["output_layers"] == [1]
        assert configs["dino_ViM_ralo.py"]["model"]["output_layers"] == [2]
        assert configs["dino_ViM_ralo.py"]["model"]["output_scale"] == 8.0
        assert configs["dino_ViM_ralo.py"]["model"]["head_hidden"] == 256
        assert configs["dino_ViM_ralo_512.py"]["model"]["head_hidden"] == 512
    finally:
        if previous is None:
            os.environ.pop("HYSMAMBA_DATA_ROOT", None)
        else:
            os.environ["HYSMAMBA_DATA_ROOT"] = previous
    print("Portable configs and output/head settings: passed")

    config_api = runpy.run_path(str(ROOT / "src/lung_severity_assesmnt/configs.py"))
    for path in (ROOT / "configs").glob("*.py"):
        json.dumps(config_api["read_config"](str(path)))
    print("Author config loader and JSON serialization: passed")

    # Execute only the actual split function and constants; no third-party imports.
    tree = ast.parse((ROOT / "src/lung_severity_assesmnt/datasets/pkl_dataset.py").read_text())
    selected = [node for node in tree.body if
                (isinstance(node, ast.FunctionDef) and node.name == "_build_split_indices") or
                (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                 ["RAW_RALO_TEST_IDS", "RALO_AUG_TEST_IDS"] for t in node.targets))]
    namespace = {"random": random}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "ralo_split", "exec"), namespace)
    split = namespace["_build_split_indices"]
    test_ids = namespace["RALO_AUG_TEST_IDS"]
    partitions = {name: split(6128, test_ids, name, val_ratio=0.1, val_seed=42)
                  for name in ["train", "val", "test"]}
    assert {name: len(ids) for name, ids in partitions.items()} == {"train": 5071, "val": 563, "test": 494}
    assert set(partitions["train"]).isdisjoint(partitions["val"])
    assert set(partitions["train"]).isdisjoint(partitions["test"])
    assert set(partitions["val"]).isdisjoint(partitions["test"])
    assert set().union(*(set(ids) for ids in partitions.values())) == set(range(6128))
    assert split(6128, test_ids, "val", 0.1, 42) == partitions["val"]
    assert split(6128, test_ids, "val", 0.1, 123) != partitions["val"]
    print("RALO row partitions and validation seed: passed (5071 / 563 / 494)")

    for path in ROOT.rglob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if re.match(r"\w+://", link) or link.startswith("#"):
                continue
            assert (path.parent / link.split("#", 1)[0]).exists(), (path, link)
    print("README/documentation local links: passed")
    print("Source validation complete. CUDA execution and training require the target environment.")


if __name__ == "__main__":
    validate()

