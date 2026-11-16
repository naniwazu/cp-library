from collections.abc import Callable


def ternary_search(lo: int, hi: int, f: Callable[[int], float]) -> int:
    """整数の区間 [lo, hi) で f を最小にする x のうち最小のものを返す（lo < hi）。

    凸関数（f(x + 1) - f(x) が広義単調増加）の最小点を求める問題に使う。
    より一般に、f が狭義単調減少してから広義単調増加に転じればよい。
    三分探索の代わりに f(x) <= f(x + 1) となる最小の x を二分探索するので、
    平らな部分があっても正しく、f の呼び出しは 2 log2(hi - lo) 回ほど
    （区間を 2/3 ずつ縮める三分探索の 6 割）。f は [lo, hi) の外では呼ばれない。
    最大点を求めるには符号を反転した関数を渡す。

    Examples:
        >>> ternary_search(-10, 10, lambda x: (x - 3) ** 2)
        3
        >>> ternary_search(0, 10, lambda x: max(abs(x - 4) - 2, 0))  # 2..6 で最小
        2
    """
    assert lo < hi
    # f(ng) > f(ng + 1) と f(ok) <= f(ok + 1) を保つ（ng = lo - 1, ok = hi - 1 は番兵）
    ng = lo - 1
    ok = hi - 1
    while ok - ng > 1:
        mid = (ok + ng) // 2
        if f(mid) <= f(mid + 1):
            ok = mid
        else:
            ng = mid
    return ok
