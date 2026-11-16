from collections.abc import Callable
from typing import Generic, TypeVar

_SparseTableS = TypeVar("_SparseTableS")


class SparseTable(Generic[_SparseTableS]):
    """更新のない列の区間積を、O(n log n) の前計算のあと O(1) で求める。

    op は結合的かつ冪等（op(x, x) == x）な二項演算とする。min / max / gcd / and /
    or などの区間クエリが多く、値を書き換えない問題で使う。更新があるなら SegTree
    を使う。op の呼び出しは構築で O(n log n) 回、prod で 1 回。

    Examples:
        >>> st = SparseTable(min, [3, 1, 4, 1, 5])
        >>> st.prod(0, 1), st.prod(2, 5), st.prod(0, 5)
        (3, 1, 1)
        >>> from math import gcd
        >>> SparseTable(gcd, [12, 18, 8]).prod(0, 2)
        6
    """

    def __init__(
        self,
        op: Callable[[_SparseTableS, _SparseTableS], _SparseTableS],
        v: list[_SparseTableS],
    ) -> None:
        # t[k][i] = op(v[i], ..., v[i + 2^k - 1])
        t = [v.copy()]
        h = 1
        while 2 * h <= len(v):
            p = t[-1]
            # 内包表記のほうが list(map(op, p, p[h:])) より PyPy で 2.5 倍ほど速い
            t.append([op(p[i], p[i + h]) for i in range(len(p) - h)])
            h *= 2
        self._op = op
        self._t = t

    def prod(self, l: int, r: int) -> _SparseTableS:
        """op(v[l], ..., v[r - 1]) を返す。l < r であること。O(1)。"""
        k = (r - l).bit_length() - 1
        t = self._t[k]
        return self._op(t[l], t[r - (1 << k)])
