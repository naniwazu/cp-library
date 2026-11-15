from collections.abc import Sequence

from cplib.ds.fenwick_tree import FenwickTree


def inversion_number(a: Sequence[int]) -> int:
    """i < j かつ a[i] > a[j] となる組の個数（転倒数）を O(n log n) で返す。

    隣接要素の交換で整列するときの最小交換回数を求める問題などに使う。

    Examples:
        >>> inversion_number([3, 1, 2, 1])
        4
    """
    rank = {x: i for i, x in enumerate(sorted(set(a)))}
    ft = FenwickTree(len(rank))
    res = 0
    for i, x in enumerate(a):
        r = rank[x]
        # 先に出た要素のうち x より大きいもの
        res += i - ft.sum(0, r + 1)
        ft.add(r, 1)
    return res
