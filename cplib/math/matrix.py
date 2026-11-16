from collections.abc import Sequence


def mat_mul(
    a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], mod: int
) -> list[list[int]]:
    """行列の積 a b の各要素を mod で割った余りを返す。a が n×m、b が m×p なら O(nmp)。

    遷移が線形な DP を行列の積でまとめる問題に使う。k 乗が必要なら mat_pow を使う。

    Examples:
        >>> mat_mul([[1, 2], [3, 4]], [[5], [6]], 10**9 + 7)
        [[17], [39]]
    """
    assert not a or len(a[0]) == len(b)
    p = len(b[0]) if b else 0
    res: list[list[int]] = []
    for ai in a:
        ci = [0] * p
        for k, bk in enumerate(b):
            aik = ai[k]
            for j, y in enumerate(bk):
                ci[j] += aik * y
            # mod <= 2^30 で要素が mod 未満なら 8 項の和まで 64 bit 整数に収まり、
            # PyPy で多倍長演算にならない。項ごとに剰余を取るより n = 200 で
            # 1.7 倍ほど速い
            if k & 7 == 7:
                ci = [x % mod for x in ci]
        res.append([x % mod for x in ci])
    return res


def mat_pow(a: Sequence[Sequence[int]], k: int, mod: int) -> list[list[int]]:
    """正方行列 a の k 乗の各要素を mod で割った余りを返す（0 <= k）。

    線形漸化式の第 k 項や、同じ線形な遷移を k 回くり返す DP を求める問題に使う。
    0 乗は単位行列。a が n×n なら O(n^3 log k)。

    Examples:
        >>> mat_pow([[1, 1], [1, 0]], 10, 10**9 + 7)  # フィボナッチ数
        [[89, 55], [55, 34]]
        >>> mat_pow([[2, 3], [4, 5]], 0, 10**9 + 7)
        [[1, 0], [0, 1]]
    """
    n = len(a)
    assert k >= 0 and all(len(row) == n for row in a)
    res = [[int(i == j) % mod for j in range(n)] for i in range(n)]
    while k:
        if k & 1:
            res = mat_mul(res, a, mod)
        k >>= 1
        if k:
            a = mat_mul(a, a, mod)
    return res
