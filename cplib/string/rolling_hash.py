from collections.abc import Sequence
from random import randrange

_RH_MOD = (1 << 61) - 1
_RH_MASK30 = (1 << 30) - 1
_RH_MASK31 = (1 << 31) - 1
# 異なる RollingHash 間でハッシュ値を比べられるよう、基数は全インスタンスで共有する。
# 固定の基数だと衝突する入力を狙って作れるので、起動ごとに乱択する
_RH_BASE = randrange(1 << 40, _RH_MOD)
_rh_pow = [1]


def _rh_mul(a: int, b: int) -> int:
    # a * b を 2^61 - 1 で割った余りと合同な 2^63 未満の値を返す。
    # PyPy で 64 bit を超える多倍長整数の演算を避けるため分割して掛ける
    au = a >> 31
    ad = a & _RH_MASK31
    bu = b >> 31
    bd = b & _RH_MASK31
    mid = ad * bu + au * bd
    return 2 * au * bu + (mid >> 30) + ((mid & _RH_MASK30) << 31) + ad * bd


def _rh_mod(x: int) -> int:
    x = (x >> 61) + (x & _RH_MOD)
    return x - _RH_MOD if x >= _RH_MOD else x


class RollingHash:
    """列 s の任意の連続部分列のハッシュ値を O(1) で返す（mod 2^61 - 1、前計算 O(n)）。

    部分文字列どうしの一致判定を高速に行いたい問題（回文判定、二分探索と組み合わせた
    最長共通接頭辞など）に使う。別の列から作った RollingHash とも比較できる。
    s の要素は str の文字か、0 以上 2^61 - 1 未満の整数。
    比べるのは同じ長さの区間どうしに限る。

    Examples:
        >>> s = RollingHash("abcabc")
        >>> s.get(0, 3) == s.get(3, 6), s.get(0, 3) == s.get(1, 4)
        (True, False)
        >>> t = RollingHash([98, 99])
        >>> t.get(0, 2) == s.get(1, 3)
        True
    """

    def __init__(self, s: str | Sequence[int]) -> None:
        a = [ord(c) for c in s] if isinstance(s, str) else s
        base = _RH_BASE
        h = [0] * (len(a) + 1)
        for i, x in enumerate(a):
            h[i + 1] = _rh_mod(_rh_mul(h[i], base) + x)
        self._hash = h
        pw = _rh_pow
        for _ in range(len(pw), len(a) + 1):
            pw.append(_rh_mod(_rh_mul(pw[-1], base)))

    def get(self, l: int, r: int) -> int:
        """s[l:r] のハッシュ値を返す。O(1)。"""
        h = self._hash
        return _rh_mod(h[r] + 4 * _RH_MOD - _rh_mul(h[l], _rh_pow[r - l]))
