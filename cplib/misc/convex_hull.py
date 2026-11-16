from collections.abc import Sequence


def convex_hull(points: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """整数座標の点の凸包の頂点を、反時計回りに返す（Andrew の monotone chain）。

    凸包の周長・面積や、凸包上の点だけを調べればよい最遠点対などの問題に使う。
    辺の途中にある点（同一直線上の点）と重複する点は含めない。始点は x が最小の
    うち y が最小の点。点が 1 種類なら 1 点、全点が同一直線上なら両端の 2 点を返す。
    O(n log n)。

    Examples:
        >>> convex_hull([(0, 0), (2, 0), (1, 1), (2, 2), (0, 2), (1, 0), (0, 0)])
        [(0, 0), (2, 0), (2, 2), (0, 2)]
        >>> convex_hull([(3, 3), (1, 1), (2, 2)]), convex_hull([(5, 5), (5, 5)])
        ([(1, 1), (3, 3)], [(5, 5)])
    """
    if not points:
        return []
    # 点を 1 つの整数に詰めてソートすると、タプルのソートより PyPy で 4 倍ほど速い
    lo = min(y for _, y in points)
    w = max(y for _, y in points) - lo + 1
    keys = sorted({x * w + y - lo for x, y in points})
    xs = [key // w for key in keys]
    ys = [key % w + lo for key in keys]
    n = len(keys)
    if n <= 2:
        return list(zip(xs, ys, strict=True))
    # h[:k] が作りかけの凸包。下側を左から、上側を右から作る
    h = [0] * (2 * n)
    k = 0
    for i in range(n):
        x = xs[i]
        y = ys[i]
        while k >= 2:
            a = h[k - 2]
            b = h[k - 1]
            if (xs[b] - xs[a]) * (y - ys[a]) - (ys[b] - ys[a]) * (x - xs[a]) > 0:
                break
            k -= 1
        h[k] = i
        k += 1
    t = k + 1
    for i in range(n - 2, -1, -1):
        x = xs[i]
        y = ys[i]
        while k >= t:
            a = h[k - 2]
            b = h[k - 1]
            if (xs[b] - xs[a]) * (y - ys[a]) - (ys[b] - ys[a]) * (x - xs[a]) > 0:
                break
            k -= 1
        h[k] = i
        k += 1
    # 最後に始点へ戻ってくるので除く
    return [(xs[h[i]], ys[h[i]]) for i in range(k - 1)]
