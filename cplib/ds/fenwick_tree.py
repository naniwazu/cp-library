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
