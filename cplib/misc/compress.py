from collections.abc import Sequence


def compress(a: Sequence[int]) -> tuple[list[int], list[int]]:
    """a を座標圧縮し、相異なる値の昇順リスト vals と各 a[i] の vals での位置を返す。

    値の範囲は広いが種類は少ない列を、配列の添字や FenwickTree の位置として
    扱う問題に使う。
    戻り値を (vals, rank) とすると vals[rank[i]] == a[i] で、
    a にない値 x の位置は bisect_left(vals, x) で求められる。
    O(n log n)。

    Examples:
        >>> compress([10, -5, 10, 3])
        ([-5, 3, 10], [2, 0, 2, 1])
    """
    vals = sorted(set(a))
    pos = {x: i for i, x in enumerate(vals)}
    return (vals, [pos[x] for x in a])
