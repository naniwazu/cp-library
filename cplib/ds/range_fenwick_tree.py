class RangeFenwickTree:
    """長さ n の整数列（初期値 0）の区間加算と区間和の取得を O(log n) で行う。

    区間にまとめて値を足しながら区間和を求める問題で使う。LazySegTree でも同じことが
    できるが、こちらは値を 1 つの int に詰める必要がなく、PyPy で数倍速い。
    FenwickTree を 2 本持ち、a[0] + ... + a[p - 1] を c0 + c1 * p の形で求める。

    Examples:
        >>> ft = RangeFenwickTree(5)
        >>> ft.add(1, 4, 2)  # [0, 2, 2, 2, 0]
        >>> ft.add(0, 2, -1)  # [-1, 1, 2, 2, 0]
        >>> ft.sum(0, 5), ft.sum(1, 3), ft.sum(4, 5)
        (4, 3, 0)
    """

    def __init__(self, n: int) -> None:
        self._n = n
        # 1-indexed で持つ（[0] は使わない）。_d0 が定数項、_d1 が p の係数
        self._d0 = [0] * (n + 1)
        self._d1 = [0] * (n + 1)

    def add(self, l: int, r: int, x: int) -> None:
        """i = l, ..., r - 1 について a[i] += x を行う。O(log n)。"""
        n = self._n
        d0 = self._d0
        d1 = self._d1
        y = x * l
        p = l + 1
        while p <= n:
            d0[p] -= y
            d1[p] += x
            p += p & -p
        y = x * r
        p = r + 1
        while p <= n:
            d0[p] += y
            d1[p] -= x
            p += p & -p

    def sum(self, l: int, r: int) -> int:
        """a[l] + ... + a[r - 1] を返す。O(log n)。"""
        d0 = self._d0
        d1 = self._d1
        s = 0
        c = 0
        p = r
        while p:
            s += d0[p]
            c += d1[p]
            p &= p - 1
        s += c * r
        c = 0
        p = l
        while p:
            s -= d0[p]
            c += d1[p]
            p &= p - 1
        return s - c * l
