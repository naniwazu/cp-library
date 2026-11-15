import ast
from collections import Counter
from pathlib import Path

import pytest

import cplib.expand
from cplib.expand import expand

PATHS = sorted(
    p
    for p in cplib.expand.ROOT.rglob("*.py")
    if p.name not in ("expand.py", "__init__.py")
)


def module_name(path: Path) -> str:
    rel = path.relative_to(cplib.expand.ROOT).with_suffix("")
    return ".".join(("cplib", *rel.parts))


@pytest.mark.parametrize("path", PATHS, ids=module_name)
def test_expanded_module_runs(path: Path) -> None:
    exec(expand(f"from {module_name(path)} import *\n"), {})


def top_level_names(path: Path) -> set[str]:
    names: set[str] = set()
    for stmt in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(stmt, ast.FunctionDef | ast.ClassDef):
            names.add(stmt.name)
        elif isinstance(stmt, ast.Assign | ast.AnnAssign):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
    return names


def test_top_level_names_are_unique_across_modules() -> None:
    # 展開後は全モジュールが 1 つの名前空間に並ぶため
    counts = Counter(name for p in PATHS for name in top_level_names(p))
    assert [name for name, c in counts.items() if c > 1] == []
