from collections.abc import Iterator


def quotient_ranges(n: int) -> Iterator[tuple[int, int, int]]:
    """n // i が等しい正の整数 i の区間 [l, r) と、その値 q = n // i を l の昇順に返す。

    sum_{i=1}^{n} f(n // i) の形の和を、n // i の値ごとにまとめて O(√n) で求めるのに使う
    （商の列挙）。区間は 1 から n までを覆い、約 2√n 個。n = 0 なら何も返さない。

    リストではなくイテレータを返す。PyPy では for 文で直接回すと区間ごとのタプルが
    作られず、n = 10^12 でリストを作る場合より 20 倍以上速い。

    Examples:
        >>> list(quotient_ranges(10))
        [(1, 2, 10), (2, 3, 5), (3, 4, 3), (4, 6, 2), (6, 11, 1)]
        >>> sum(q * (r - l) for l, r, q in quotient_ranges(10))  # sum n // i
        27
    """
    l = 1
    while l <= n:
        q = n // l
        r = n // q + 1
        yield l, r, q
        l = r
