"""`from cplib.x.y import ...` をライブラリ本体に置き換えて 1 ファイルにする。"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).parent


class ExpandError(Exception):
    pass


def _docstrings(tree: ast.Module) -> dict[int, tuple[int, str]]:
    repl: dict[int, tuple[int, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        first = node.body[0] if node.body else None
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        # 本体が docstring だけだと構文が壊れるので pass を残す
        text = " " * first.col_offset + "pass\n" if len(node.body) == 1 else ""
        repl[first.lineno - 1] = (first.end_lineno or first.lineno, text)
    return repl


def _load(module: str, done: set[str]) -> str:
    if module in done:
        return ""
    path = ROOT.joinpath(*module.split(".")[1:]).with_suffix(".py")
    if not path.is_file():
        raise ExpandError(f"module not found: {module}")
    done.add(module)
    body = _expand(path.read_text(encoding="utf-8"), done, lib=True)
    return f"# --- {module} ---\n{body}" + ("" if body.endswith("\n") else "\n")


def _expand(src: str, done: set[str], lib: bool) -> str:
    tree = ast.parse(src)
    repl = _docstrings(tree) if lib else {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name == "cplib" or alias.name.startswith("cplib."):
                    raise ExpandError(f"use 'from {alias.name} import ...'")
        if not isinstance(stmt, ast.ImportFrom) or stmt.level or not stmt.module:
            continue
        if lib and stmt.module == "__future__":
            # 展開先ではファイル先頭に置けず SyntaxError になる
            raise ExpandError("'from __future__' is not allowed in library modules")
        if not stmt.module.startswith("cplib."):
            continue
        if any(alias.asname for alias in stmt.names):
            raise ExpandError(f"'as' is not supported: {stmt.module}")
        end = stmt.end_lineno or stmt.lineno
        repl[stmt.lineno - 1] = (end, _load(stmt.module, done))

    lines = src.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        if i in repl:
            i, text = repl[i]
            out.append(text)
        else:
            out.append(lines[i])
            i += 1
    return "".join(out)


def expand(src: str) -> str:
    return _expand(src, set(), lib=False)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: cplib-expand FILE")
    try:
        print(expand(Path(sys.argv[1]).read_text(encoding="utf-8")), end="")
    except ExpandError as e:
        sys.exit(f"cplib-expand: {e}")
