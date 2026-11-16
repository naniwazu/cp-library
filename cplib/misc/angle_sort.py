from collections.abc import Sequence


def argsort_by_angle(points: Sequence[tuple[int, int]]) -> list[int]:
    """原点から見た偏角の昇順に点の添字を並べて返す（偏角ソート）。

    原点のまわりを回りながら点やベクトルを順に調べる問題（走査線の回転、
    ベクトルの和の最大化など）に使う。偏角は x 軸の正の向きを 0 とし、反時計回りに
    [0, 2π) で測る。偏角が等しい点どうしは添字の昇順（安定）。整数だけで比較するので
    誤差はない。原点 (0, 0) は含まないこと。O(n log n)。

    座標の絶対値が 10^6、n が 2*10^5 程度までなら途中の値が 64 ビットに収まり、
    比較関数を cmp_to_key で渡すソートより PyPy で 15 倍ほど速い。座標が 10^9 程度だと
    多倍長になり、差は 3 倍ほどに縮む。

    Examples:
        >>> ps = [(-1, 0), (1, 1), (0, -1), (1, 0), (2, 2), (-1, 1)]
        >>> argsort_by_angle(ps)
        [3, 1, 4, 5, 0, 2]
    """
    n = len(points)
    c = max((max(abs(x), abs(y)) for x, y in points), default=0)
    # 象限ごとに 90 度ずつ回して (u, v) (u > 0, v >= 0) に移すと、象限内の偏角は
    # v / (u + v) について単調。u + v <= 2c なので、相異なる値の差は 1 / (4c^2) 以上で、
    # 2^k >= 4c^2 倍して切り捨てても順序が保たれる
    k = (4 * c * c).bit_length()
    keys: list[int] = []
    for i, (x, y) in enumerate(points):
        if x > 0 and y >= 0:
            q, u, v = 0, x, y
        elif x <= 0 and y > 0:
            q, u, v = 1, y, -x
        elif x < 0 and y <= 0:
            q, u, v = 2, -x, -y
        else:
            assert y < 0, "原点は偏角が定まらない"
            q, u, v = 3, -y, x
        keys.append(((q << k) | (v << k) // (u + v)) * n + i)
    keys.sort()
    return [key % n for key in keys]
