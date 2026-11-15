from pathlib import Path

import pytest

import cplib.expand
from cplib.expand import ExpandError, expand


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cplib.expand, "ROOT", tmp_path)
    (tmp_path / "a").mkdir()
    return tmp_path


def write(root: Path, name: str, src: str) -> None:
    (root / "a" / f"{name}.py").write_text(src, encoding="utf-8")


def test_expands_dependencies_once_before_use(root: Path) -> None:
    write(root, "base", "def base() -> int:\n    return 1\n")
    write(
        root,
        "mid",
        "from cplib.a.base import base\n\n\ndef mid() -> int:\n    return base() + 1\n",
    )
    src = (
        "from cplib.a.mid import mid\n"
        "from cplib.a.base import base\n"
        "\nans = (mid(), base())\n"
    )

    out = expand(src)

    assert out == (
        "# --- cplib.a.mid ---\n"
        "# --- cplib.a.base ---\n"
        "def base() -> int:\n    return 1\n"
        "\n\ndef mid() -> int:\n    return base() + 1\n"
        "\nans = (mid(), base())\n"
    )
    ns: dict[str, object] = {}
    exec(out, ns)
    assert ns["ans"] == (2, 1)


def test_strips_library_docstrings_only(root: Path) -> None:
    write(
        root,
        "doc",
        '"""module"""\n\n\nclass C:\n    """class"""\n\n    def f(self) -> None:\n'
        '        """only a docstring"""\n',
    )
    src = '"""user"""\nfrom cplib.a.doc import C\n\nC().f()\n'

    out = expand(src)

    assert out == (
        '"""user"""\n'
        "# --- cplib.a.doc ---\n"
        "\n\nclass C:\n\n    def f(self) -> None:\n        pass\n"
        "\nC().f()\n"
    )
    exec(out, {})


def test_star_import(root: Path) -> None:
    write(root, "base", "X = 1\n")
    ns: dict[str, object] = {}
    exec(expand("from cplib.a.base import *\n"), ns)
    assert ns["X"] == 1


@pytest.mark.parametrize(
    ("lib", "src"),
    [
        ("X = 1\n", "import cplib.a.base\n"),
        ("X = 1\n", "from cplib.a.base import X as Y\n"),
        ("X = 1\n", "from cplib.a.missing import X\n"),
        ("from __future__ import annotations\nX = 1\n", "from cplib.a.base import X\n"),
    ],
    ids=["import", "alias", "missing", "future"],
)
def test_errors(root: Path, lib: str, src: str) -> None:
    write(root, "base", lib)
    with pytest.raises(ExpandError):
        expand(src)
