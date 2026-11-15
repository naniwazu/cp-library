from bisect import bisect_left
from collections.abc import Sequence


def lis(a: Sequence[int]) -> int:
    """狭義単調増加な部分列の最長の長さを O(n log n) で返す。

    広義単調増加で求めたいときは、bisect_left を bisect_right に替えたものを使う。

    Examples:
        >>> lis([3, 1, 4, 1, 5, 9, 2, 6])
        4
    """
    tails: list[int] = []
    for x in a:
        i = bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)
