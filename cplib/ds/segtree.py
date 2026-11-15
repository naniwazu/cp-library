from collections.abc import Callable
from typing import Generic, TypeVar

_SegTreeS = TypeVar("_SegTreeS")


class SegTree(Generic[_SegTreeS]):
    """モノイドの列に対する 1 点更新と区間積の取得を O(log n) で行う。

    op は結合的な二項演算、e はその単位元とする。要素を 1 点ずつ書き換えながら
    区間和・区間最小・区間の行列積などを求める問題で使う。max_right / min_left で
    「条件を満たす最長の区間」を二分探索できる。ac-library-python の SegTree と
    同じ API。構築は O(n)、各操作は op を O(log n) 回呼ぶ。

    Examples:
        >>> st = SegTree(lambda a, b: a + b, 0, [3, 1, 4, 1, 5])
        >>> st.prod(1, 4)
        6
        >>> st.set(2, 0)
        >>> st.get(2), st.all_prod()
        (0, 10)
    """

    def __init__(
        self,
        op: Callable[[_SegTreeS, _SegTreeS], _SegTreeS],
        e: _SegTreeS,
        v: int | list[_SegTreeS],
    ) -> None:
        if isinstance(v, int):
            v = [e] * v
        n = len(v)
        size = 1
        while size < n:
            size *= 2
        # d[1] が根、d[size + i] が a[i]
        d = [e] * size + v + [e] * (size - n)
        for i in range(size - 1, 0, -1):
            d[i] = op(d[2 * i], d[2 * i + 1])
        self._op = op
        self._e = e
        self._n = n
        self._size = size
        self._d = d

    def set(self, p: int, x: _SegTreeS) -> None:
        """a[p] = x を行う。O(log n)。"""
        op = self._op
        d = self._d
        p += self._size
        d[p] = x
        while p > 1:
            p >>= 1
            d[p] = op(d[2 * p], d[2 * p + 1])

    def get(self, p: int) -> _SegTreeS:
        """a[p] を返す。O(1)。"""
        return self._d[p + self._size]

    def prod(self, l: int, r: int) -> _SegTreeS:
        """op(a[l], ..., a[r - 1]) を返す。l == r なら e。O(log n)。"""
        op = self._op
        d = self._d
        sml = smr = self._e
        l += self._size
        r += self._size
        while l < r:
            if l & 1:
                sml = op(sml, d[l])
                l += 1
            if r & 1:
                r -= 1
                smr = op(d[r], smr)
            l >>= 1
            r >>= 1
        return op(sml, smr)

    def all_prod(self) -> _SegTreeS:
        """op(a[0], ..., a[n - 1]) を O(1) で返す。"""
        return self._d[1]

    def max_right(self, l: int, f: Callable[[_SegTreeS], bool]) -> int:
        """f(prod(l, r)) が真になる最大の r（l <= r <= n）を返す。O(log n)。

        f(e) は真であること。正確には f(prod(l, r)) が真で、r == n か
        f(prod(l, r + 1)) が偽であるような r を 1 つ返す。f が単調（区間を右へ
        伸ばすと真から偽にしか変わらない）なら、それは上の最大の r になる。

        Examples:
            >>> st = SegTree(lambda a, b: a + b, 0, [3, 1, 4, 1, 5])
            >>> st.max_right(1, lambda s: s <= 5)  # 1 + 4 <= 5 < 1 + 4 + 1
            3
            >>> st.max_right(0, lambda s: s <= 100)
            5
            >>> st.max_right(0, lambda s: s <= 2)  # a[0] だけで超える
            0
        """
        n = self._n
        if l == n:
            return n
        op = self._op
        d = self._d
        size = self._size
        l += size
        sm = self._e
        while True:
            while not l & 1:
                l >>= 1
            t = op(sm, d[l])
            if not f(t):
                # 節点 l の中で境界を探して葉まで下りる
                while l < size:
                    l *= 2
                    t = op(sm, d[l])
                    if f(t):
                        sm = t
                        l += 1
                return l - size
            sm = t
            l += 1
            if l & -l == l:
                return n

    def min_left(self, r: int, f: Callable[[_SegTreeS], bool]) -> int:
        """f(prod(l, r)) が真になる最小の l（0 <= l <= r）を返す。O(log n)。

        f(e) は真であること。正確には f(prod(l, r)) が真で、l == 0 か
        f(prod(l - 1, r)) が偽であるような l を 1 つ返す。f が単調（区間を左へ
        伸ばすと真から偽にしか変わらない）なら、それは上の最小の l になる。

        Examples:
            >>> st = SegTree(lambda a, b: a + b, 0, [3, 1, 4, 1, 5])
            >>> st.min_left(5, lambda s: s <= 6)  # 1 + 5 <= 6 < 4 + 1 + 5
            3
            >>> st.min_left(5, lambda s: s <= 100)
            0
        """
        if r == 0:
            return 0
        op = self._op
        d = self._d
        size = self._size
        r += size
        sm = self._e
        while True:
            r -= 1
            while r > 1 and r & 1:
                r >>= 1
            t = op(d[r], sm)
            if not f(t):
                # 節点 r の中で境界を探して葉まで下りる
                while r < size:
                    r = 2 * r + 1
                    t = op(d[r], sm)
                    if f(t):
                        sm = t
                        r -= 1
                return r + 1 - size
            sm = t
            if r & -r == r:
                return 0
