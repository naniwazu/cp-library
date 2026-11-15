from collections.abc import Callable


def binary_search(ok: int, ng: int, check: Callable[[int], bool]) -> int:
    """check(ok) が真、check(ng) が偽のとき、境界で真になる側の整数を返す。

    check が ok 側で真、ng 側で偽に単調に切り替わる「答えで二分探索」に使う。
    ok < ng でも ok > ng でもよく、check は端点では呼ばれない。
    check の呼び出しは O(log |ok - ng|) 回。

    Examples:
        >>> binary_search(0, 100, lambda x: x * x <= 50)
        7
        >>> binary_search(100, 0, lambda x: x * x >= 50)
        8
    """
    while abs(ok - ng) > 1:
        mid = (ok + ng) // 2
        if check(mid):
            ok = mid
        else:
            ng = mid
    return ok
