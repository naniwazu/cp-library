from collections.abc import Sequence
from math import isqrt


def mo_order(n: int, queries: Sequence[tuple[int, int]]) -> list[int]:
    """Mo's algorithm で区間クエリを処理する順序（クエリの添字の列）を返す。

    長さ n の列への区間 [l, r) のクエリ（0 <= l <= r <= n）がオフラインで与えられ、
    区間の端を 1 つ伸縮したときの答えの更新が速い問題（区間の種類数など）に使う。
    l をブロックに分け、ブロックごとに r の向きを交互に反転して並べる。
    クエリ数を q として、並べ替えは O(q log q)、伸縮の回数は O(n √q)。

    伸縮のループは呼び出し側に書く（追加・削除を関数にして渡す形より PyPy で
    1.2〜1.4 倍速い）::

        l = r = 0
        for i in mo_order(n, queries):
            ql, qr = queries[i]
            while l > ql:
                l -= 1
                # a[l] を追加
            while r < qr:
                # a[r] を追加
                r += 1
            while l < ql:
                # a[l] を削除
                l += 1
            while r > qr:
                r -= 1
                # a[r] を削除
            # ans[i] = 現在の答え

    Examples:
        >>> mo_order(8, [(0, 8), (1, 2), (0, 3), (5, 6)])
        [1, 2, 3, 0]
    """
    q = len(queries)
    if not q:
        return []
    # 左端の移動が平均でブロック幅の 1/3、右端の移動がブロックあたり n 程度なので、
    # 合計 q b / 3 + n^2 / b を最小にする b = n √(3 / q) を使う。n = q = 2*10^5 の
    # 計測でも b = n / √q より 2 割速かった
    b = max(1, isqrt(3 * n * n // q))
    keys: list[int] = []
    for i, (l, r) in enumerate(queries):
        k = l // b
        keys.append((k * (n + 1) + (n - r if k & 1 else r)) * q + i)
    keys.sort()
    return [key % q for key in keys]
