from collections.abc import Iterable
from itertools import groupby
from typing import TypeVar

_RunLengthT = TypeVar("_RunLengthT")


def run_length(a: Iterable[_RunLengthT]) -> list[tuple[_RunLengthT, int]]:
    """a を等しい値が続く部分ごとに (値, 個数) にまとめたリストを返す。O(n)。

    ランレングス圧縮。文字列で同じ文字が続く部分の長さを扱う問題などに使う。
    文字列なら値は 1 文字の str になる。

    Examples:
        >>> run_length("aaabccdd")
        [('a', 3), ('b', 1), ('c', 2), ('d', 2)]
        >>> run_length([1, 1, 2, 1])
        [(1, 2), (2, 1), (1, 1)]
    """
    return [(x, len(list(g))) for x, g in groupby(a)]
