class FenwickTree:
    """長さ n の整数列（初期値 0）の 1 点加算と区間和の取得を O(log n) で行う。

    転倒数の計算や、座標圧縮した値の個数を数えながら走査する問題で使う。

    Examples:
        >>> ft = FenwickTree(5)
        >>> ft.add(1, 3)
        >>> ft.add(3, 4)
        >>> ft.sum(0, 3)
        3
        >>> ft.sum(1, 5)
        7
    """

    def __init__(self, n: int) -> None:
        self._n = n
        # 1-indexed で持つ（_data[0] は使わない）
        self._data = [0] * (n + 1)

    def add(self, p: int, x: int) -> None:
        """a[p] += x を行う。O(log n)。"""
        n = self._n
        data = self._data
        p += 1
        while p <= n:
            data[p] += x
            p += p & -p

    def sum(self, l: int, r: int) -> int:
        """a[l] + ... + a[r - 1] を返す。O(log n)。"""
        data = self._data
        s = 0
        while r:
            s += data[r]
            r &= r - 1
        while l:
            s -= data[l]
            l &= l - 1
        return s

    def bisect_left(self, w: int) -> int:
        """a[0] + ... + a[i] >= w となる最小の i を返す。なければ n。O(log n)。

        各 a[i] は 0 以上であること。累積和の列 [a[0], a[0] + a[1], ...] に対する
        bisect.bisect_left と同じ値になる。a[v] を値 v の個数とすれば、k 番目
        （0-indexed）に小さい値は bisect_left(k + 1) で求まる。

        Examples:
            >>> ft = FenwickTree(6)
            >>> for v in [4, 1, 4, 2]:  # 値 v の個数を数える
            ...     ft.add(v, 1)
            >>> [ft.bisect_left(k + 1) for k in range(4)]  # 小さいほうから k 番目
            [1, 2, 4, 4]
            >>> ft.bisect_left(5), ft.bisect_left(0)
            (6, 0)
        """
        n = self._n
        data = self._data
        p = 0
        # 2 のべきの幅で、和が w 未満のまま進める最も右の位置まで下りる
        k = 1 << n.bit_length()
        while k:
            if p + k <= n and data[p + k] < w:
                p += k
                w -= data[p]
            k >>= 1
        return p
