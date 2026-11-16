from collections.abc import Callable
from typing import Generic, TypeVar

_LazySegTreeS = TypeVar("_LazySegTreeS")
_LazySegTreeF = TypeVar("_LazySegTreeF")


class LazySegTree(Generic[_LazySegTreeS, _LazySegTreeF]):
    """モノイドの列に対する区間への写像の作用と区間積の取得を O(log n) で行う。

    区間加算・区間最小や、区間 affine 変換・区間和のように、区間をまとめて書き換え
    ながら区間をまとめた値を求める問題で使う。ac-library-python の LazySegTree と
    同じ API で、各操作は op などを O(log n) 回呼ぶ。構築は O(n)。

    - op, e: 値のモノイド（SegTree と同じ）。
    - mapping(f, x): 写像 f を値 x に作用させた結果。
      mapping(f, op(x, y)) == op(mapping(f, x), mapping(f, y)) であること。
    - composition(f, g): g の後に f を作用させる写像 f∘g。
    - id_: 恒等写像。

    ac-library-python と違い、apply(p, f) という 1 点への作用の形は持たない
    （引数の数で意味を変えると型が付けにくく、遅くもなるため）。1 点には
    apply(p, p + 1, f) を使う。

    値や写像をタプルで持つと生成コストで数倍遅くなる。区間 affine・区間和のように
    複数の数を持つ場合は、`v << 20 | 長さ` のように 1 つの int に詰める。
    ただし詰めた値が 2^63 を超えると多倍長になり、かえって遅くなる。

    Examples:
        区間加算・区間最小（mapping も composition も加算、id_ は 0）:

        >>> add = lambda a, b: a + b
        >>> st = LazySegTree(min, 10**18, add, add, 0, [5, 2, 6, 3, 7])
        >>> st.apply(0, 3, 10)  # [15, 12, 16, 3, 7]
        >>> st.prod(0, 3), st.prod(2, 5)
        (12, 3)
        >>> st.get(1)
        12
    """

    def __init__(
        self,
        op: Callable[[_LazySegTreeS, _LazySegTreeS], _LazySegTreeS],
        e: _LazySegTreeS,
        mapping: Callable[[_LazySegTreeF, _LazySegTreeS], _LazySegTreeS],
        composition: Callable[[_LazySegTreeF, _LazySegTreeF], _LazySegTreeF],
        id_: _LazySegTreeF,
        v: int | list[_LazySegTreeS],
    ) -> None:
        if isinstance(v, int):
            v = [e] * v
        n = len(v)
        log = 0
        while (1 << log) < n:
            log += 1
        size = 1 << log
        # d[1] が根、d[size + i] が a[i]。lz[k] は節点 k の子へまだ伝えていない写像
        d = [e] * size + v + [e] * (size - n)
        for i in range(size - 1, 0, -1):
            d[i] = op(d[2 * i], d[2 * i + 1])
        self._op = op
        self._e = e
        self._mapping = mapping
        self._composition = composition
        self._id = id_
        self._n = n
        self._log = log
        self._size = size
        self._d = d
        self._lz = [id_] * size

    def set(self, p: int, x: _LazySegTreeS) -> None:
        """a[p] = x を行う。O(log n)。"""
        op = self._op
        d = self._d
        p += self._size
        for i in range(self._log, 0, -1):
            self._push(p >> i)
        d[p] = x
        while p > 1:
            p >>= 1
            d[p] = op(d[2 * p], d[2 * p + 1])

    def get(self, p: int) -> _LazySegTreeS:
        """a[p] を返す。O(log n)。"""
        p += self._size
        for i in range(self._log, 0, -1):
            self._push(p >> i)
        return self._d[p]

    def prod(self, l: int, r: int) -> _LazySegTreeS:
        """op(a[l], ..., a[r - 1]) を返す。l == r なら e。O(log n)。"""
        if l == r:
            return self._e
        op = self._op
        d = self._d
        l += self._size
        r += self._size
        for i in range(self._log, 0, -1):
            if (l >> i) << i != l:
                self._push(l >> i)
            if (r >> i) << i != r:
                self._push((r - 1) >> i)
        sml = smr = self._e
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

    def all_prod(self) -> _LazySegTreeS:
        """op(a[0], ..., a[n - 1]) を O(1) で返す。"""
        return self._d[1]

    def apply(self, l: int, r: int, f: _LazySegTreeF) -> None:
        """i = l, ..., r - 1 について a[i] = mapping(f, a[i]) を行う。O(log n)。"""
        if l == r:
            return
        op = self._op
        mapping = self._mapping
        composition = self._composition
        d = self._d
        lz = self._lz
        log = self._log
        size = self._size
        l += size
        r += size
        for i in range(log, 0, -1):
            if (l >> i) << i != l:
                self._push(l >> i)
            if (r >> i) << i != r:
                self._push((r - 1) >> i)
        l0, r0 = l, r
        while l < r:
            if l & 1:
                d[l] = mapping(f, d[l])
                if l < size:
                    lz[l] = composition(f, lz[l])
                l += 1
            if r & 1:
                r -= 1
                d[r] = mapping(f, d[r])
                if r < size:
                    lz[r] = composition(f, lz[r])
            l >>= 1
            r >>= 1
        for i in range(1, log + 1):
            if (l0 >> i) << i != l0:
                k = l0 >> i
                d[k] = op(d[2 * k], d[2 * k + 1])
            if (r0 >> i) << i != r0:
                k = (r0 - 1) >> i
                d[k] = op(d[2 * k], d[2 * k + 1])

    def max_right(self, l: int, g: Callable[[_LazySegTreeS], bool]) -> int:
        """g(prod(l, r)) が真になる最大の r（l <= r <= n）を返す。O(log n)。

        意味は SegTree.max_right と同じ。

        Examples:
            >>> add = lambda a, b: a + b
            >>> st = LazySegTree(min, 10**18, add, add, 0, [5, 2, 6, 3, 7])
            >>> st.apply(1, 2, 10)  # [5, 12, 6, 3, 7]
            >>> st.max_right(0, lambda x: x >= 5)  # min(5, 12, 6) >= 5 > 3
            3
        """
        n = self._n
        if l == n:
            return n
        op = self._op
        d = self._d
        size = self._size
        l += size
        for i in range(self._log, 0, -1):
            self._push(l >> i)
        sm = self._e
        while True:
            while not l & 1:
                l >>= 1
            t = op(sm, d[l])
            if not g(t):
                while l < size:
                    self._push(l)
                    l *= 2
                    t = op(sm, d[l])
                    if g(t):
                        sm = t
                        l += 1
                return l - size
            sm = t
            l += 1
            if l & -l == l:
                return n

    def min_left(self, r: int, g: Callable[[_LazySegTreeS], bool]) -> int:
        """g(prod(l, r)) が真になる最小の l（0 <= l <= r）を返す。O(log n)。

        意味は SegTree.min_left と同じ。

        Examples:
            >>> add = lambda a, b: a + b
            >>> st = LazySegTree(min, 10**18, add, add, 0, [5, 2, 6, 3, 7])
            >>> st.apply(3, 4, 10)  # [5, 2, 6, 13, 7]
            >>> st.min_left(5, lambda x: x >= 5)  # 2 < 5 <= min(6, 13, 7)
            2
        """
        if r == 0:
            return 0
        op = self._op
        d = self._d
        size = self._size
        r += size
        for i in range(self._log, 0, -1):
            self._push((r - 1) >> i)
        sm = self._e
        while True:
            r -= 1
            while r > 1 and r & 1:
                r >>= 1
            t = op(d[r], sm)
            if not g(t):
                while r < size:
                    self._push(r)
                    r = 2 * r + 1
                    t = op(d[r], sm)
                    if g(t):
                        sm = t
                        r -= 1
                return r + 1 - size
            sm = t
            if r & -r == r:
                return 0

    def _push(self, k: int) -> None:
        # ループを持たない小さな関数にしておくと、PyPy の JIT が呼び出し元へ
        # インライン化する（ループを含む関数は呼び出しのまま残り遅い）
        d = self._d
        lz = self._lz
        mapping = self._mapping
        f = lz[k]
        c = 2 * k
        d[c] = mapping(f, d[c])
        d[c + 1] = mapping(f, d[c + 1])
        if c < self._size:
            composition = self._composition
            lz[c] = composition(f, lz[c])
            lz[c + 1] = composition(f, lz[c + 1])
        lz[k] = self._id
