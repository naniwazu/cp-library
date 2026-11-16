from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator
from math import isqrt
from typing import Any, Generic, Protocol, TypeVar


class _SortedListComparable(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...


_SortedListT = TypeVar("_SortedListT", bound=_SortedListComparable)


class SortedList(Generic[_SortedListT]):
    """要素の追加・削除をしながら、x 以上の最小値や k 番目の値を O(√n) で求める。

    C++ の multiset に当たる順序つき多重集合で、重複を許す。「今ある値のうち x に
    最も近いもの」「k 番目に小さいもの」を更新のたびに求める問題で使う。API は
    sortedcontainers.SortedList の一部と同じで、加えて lt / le / gt / ge を持つ。
    要素どうしは < で比較できること。構築は O(n log n)、各操作は O(√n)。

    値域が小さいか先読みできるなら、座標圧縮して FenwickTree の上で二分探索する
    ほうが速い。

    Examples:
        >>> s = SortedList([3, 1, 4, 1, 5])
        >>> s.add(9)
        >>> list(s), len(s)
        ([1, 1, 3, 4, 5, 9], 6)
        >>> s.ge(2), s.gt(5), s.lt(1), s.le(4)
        (3, 9, None, 4)
        >>> s[0], s[-1], s.bisect_left(4), s.count(1)
        (1, 9, 3, 2)
        >>> s.discard(1)
        >>> s.pop(), 1 in s, 2 in s
        (9, True, False)
    """

    def __init__(self, iterable: Iterable[_SortedListT] = ()) -> None:
        a = sorted(iterable)
        n = len(a)
        # 平方分割。バケット数を b として各バケットを 16b 個前後で作り、
        # 24b 個を超えたら 2 つに割る。b ≒ √(n / 16) に保たれる
        b = isqrt(n // 16) + 1 if n else 0
        self._a = [a[n * i // b : n * (i + 1) // b] for i in range(b)]
        # 各バケットの最大値。値の入るバケットはこれを先頭から線形に走査して探す。
        # PyPy 3.11 の bisect は純 Python 実装で、呼び出しと分岐の多いループが重く、
        # 要素数 2×10^5 以下では線形走査のほうが 1〜3 割速い。10^6（バケット数
        # 250 前後）でも整数なら二分探索と 5% 以内の差に収まる
        self._m = [x[-1] for x in self._a]
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __iter__(self) -> Iterator[_SortedListT]:
        for b in self._a:
            yield from b

    def __contains__(self, x: _SortedListT) -> bool:
        m = self._m
        if not m or m[-1] < x:
            return False
        i = 0
        while m[i] < x:
            i += 1
        b = self._a[i]
        return b[bisect_left(b, x)] == x

    def __getitem__(self, k: int) -> _SortedListT:
        """小さいほうから k 番目（0-indexed）の値を返す。負の k は末尾から数える。"""
        a = self._a
        if k < 0:
            for i in range(len(a) - 1, -1, -1):
                k += len(a[i])
                if k >= 0:
                    return a[i][k]
        else:
            for b in a:
                if k < len(b):
                    return b[k]
                k -= len(b)
        raise IndexError("SortedList index out of range")

    def add(self, x: _SortedListT) -> None:
        """x を 1 個追加する。"""
        a = self._a
        m = self._m
        self._n += 1
        if not a:
            a.append([x])
            m.append(x)
            return
        if x < m[-1]:
            i = 0
            while not x < m[i]:
                i += 1
            b = a[i]
            b.insert(bisect_right(b, x), x)
        else:
            # 最大値以上なら末尾に足す。昇順に追加し続ける場合を O(1) にするため
            i = len(m) - 1
            b = a[i]
            b.append(x)
            m[i] = x
        if len(b) > 24 * len(a):
            h = len(b) >> 1
            a.insert(i + 1, b[h:])
            del b[h:]
            m.insert(i, b[-1])

    def _pop_at(self, i: int, j: int) -> _SortedListT:
        b = self._a[i]
        x = b.pop(j)
        self._n -= 1
        if not b:
            del self._a[i]
            del self._m[i]
        elif j == len(b):
            self._m[i] = b[-1]
        return x

    def discard(self, x: _SortedListT) -> None:
        """x があれば 1 個だけ取り除く。なければ何もしない。"""
        m = self._m
        if not m or m[-1] < x:
            return
        i = 0
        while m[i] < x:
            i += 1
        b = self._a[i]
        j = bisect_left(b, x)
        if b[j] == x:
            self._pop_at(i, j)

    def remove(self, x: _SortedListT) -> None:
        """x を 1 個取り除く。なければ ValueError を送出する。"""
        n = self._n
        self.discard(x)
        if self._n == n:
            raise ValueError(f"{x!r} not in SortedList")

    def pop(self, k: int = -1) -> _SortedListT:
        """k 番目の値を取り除いて返す。既定では最大値。負の k は末尾から数える。"""
        a = self._a
        if k < 0:
            for i in range(len(a) - 1, -1, -1):
                k += len(a[i])
                if k >= 0:
                    return self._pop_at(i, k)
        else:
            for i in range(len(a)):
                if k < len(a[i]):
                    return self._pop_at(i, k)
                k -= len(a[i])
        raise IndexError("pop index out of range")

    def bisect_left(self, x: _SortedListT) -> int:
        """x 未満の要素の個数を返す。"""
        k = 0
        for b in self._a:
            if not b[-1] < x:
                return k + bisect_left(b, x)
            k += len(b)
        return k

    def bisect_right(self, x: _SortedListT) -> int:
        """x 以下の要素の個数を返す。"""
        k = 0
        for b in self._a:
            if x < b[-1]:
                return k + bisect_right(b, x)
            k += len(b)
        return k

    def index(self, x: _SortedListT) -> int:
        """x が最初に現れる位置を返す。なければ ValueError を送出する。"""
        k = self.bisect_left(x)
        if k == self._n or self[k] != x:
            raise ValueError(f"{x!r} not in SortedList")
        return k

    def count(self, x: _SortedListT) -> int:
        """x の個数を返す。"""
        return self.bisect_right(x) - self.bisect_left(x)

    def lt(self, x: _SortedListT) -> _SortedListT | None:
        """x 未満の最大の要素を返す。なければ None。"""
        m = self._m
        if not m or m[-1] < x:
            return m[-1] if m else None
        i = 0
        while m[i] < x:
            i += 1
        b = self._a[i]
        j = bisect_left(b, x)
        if j:
            return b[j - 1]
        return m[i - 1] if i else None

    def le(self, x: _SortedListT) -> _SortedListT | None:
        """x 以下の最大の要素を返す。なければ None。"""
        m = self._m
        if not m or not x < m[-1]:
            return m[-1] if m else None
        i = 0
        while not x < m[i]:
            i += 1
        b = self._a[i]
        j = bisect_right(b, x)
        if j:
            return b[j - 1]
        return m[i - 1] if i else None

    def gt(self, x: _SortedListT) -> _SortedListT | None:
        """x より大きい最小の要素を返す。なければ None。"""
        m = self._m
        if not m or not x < m[-1]:
            return None
        i = 0
        while not x < m[i]:
            i += 1
        b = self._a[i]
        return b[bisect_right(b, x)]

    def ge(self, x: _SortedListT) -> _SortedListT | None:
        """x 以上の最小の要素を返す。なければ None。"""
        m = self._m
        if not m or m[-1] < x:
            return None
        i = 0
        while m[i] < x:
            i += 1
        b = self._a[i]
        return b[bisect_left(b, x)]
