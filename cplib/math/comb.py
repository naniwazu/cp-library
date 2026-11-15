class Comb:
    """素数 mod での二項係数・順列の数・階乗とその逆元を O(1) で返す（前計算 O(n)）。

    数え上げで nCr mod p を何度も使う問題に使う。表は必要に応じて自動で伸びるので、
    最大の n を先に渡しておくと伸ばす回数が減る。引数は mod 未満であること。

    Examples:
        >>> c = Comb(998244353)
        >>> c.comb(5, 2), c.perm(5, 2), c.fac(5)
        (10, 20, 120)
        >>> c.inv(2) * 2 % 998244353
        1
        >>> c.comb(3, 4), c.comb(3, -1)
        (0, 0)
    """

    def __init__(self, mod: int, n: int = 1) -> None:
        self._mod = mod
        self._fac = [1, 1]
        self._finv = [1, 1]
        self._inv = [0, 1]
        self._extend(n)

    def _extend(self, n: int) -> None:
        mod = self._mod
        fac = self._fac
        finv = self._finv
        inv = self._inv
        m = len(fac)
        fac += [0] * (n + 1 - m)
        finv += [0] * (n + 1 - m)
        inv += [0] * (n + 1 - m)
        for i in range(m, n + 1):
            fac[i] = fac[i - 1] * i % mod
            inv[i] = mod - inv[mod % i] * (mod // i) % mod
            finv[i] = finv[i - 1] * inv[i] % mod

    def comb(self, n: int, r: int) -> int:
        """nCr を返す。r < 0 または n < r なら 0。"""
        if r < 0 or n < r:
            return 0
        if n >= len(self._fac):
            self._extend(n)
        finv = self._finv
        return self._fac[n] * finv[r] % self._mod * finv[n - r] % self._mod

    def perm(self, n: int, r: int) -> int:
        """nPr を返す。r < 0 または n < r なら 0。"""
        if r < 0 or n < r:
            return 0
        if n >= len(self._fac):
            self._extend(n)
        return self._fac[n] * self._finv[n - r] % self._mod

    def fac(self, n: int) -> int:
        """n! を返す。"""
        if n >= len(self._fac):
            self._extend(n)
        return self._fac[n]

    def finv(self, n: int) -> int:
        """1 / n! を返す。"""
        if n >= len(self._fac):
            self._extend(n)
        return self._finv[n]

    def inv(self, n: int) -> int:
        """1 / n を返す（1 <= n）。"""
        if n >= len(self._fac):
            self._extend(n)
        return self._inv[n]
