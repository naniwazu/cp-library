from heapq import heappop, heappush


def dijkstra(g: list[list[tuple[int, int]]], s: int) -> list[int]:
    """非負重みのグラフで、s から各頂点への最短距離を返す。O((n + m) log m)。

    重みつきグラフの単一始点最短経路に使う。g[v] は (行き先, コスト) のリストで、
    コストは非負。到達できない頂点の距離は -1。

    ヒープには 距離 << n.bit_length() | 頂点 の int を積む。距離がおよそ
    2^(63 - n.bit_length()) を超えると多倍長整数になり、結果は正しいまま
    2 倍ほど遅くなる（n = 5×10^5 なら距離 10^13 程度が境目）。

    Examples:
        >>> g = [[(1, 4), (2, 1)], [], [(1, 2)], []]
        >>> dijkstra(g, 0)
        [0, 3, 1, -1]
    """
    n = len(g)
    # タプルを積むより速い
    shift = n.bit_length()
    mask = (1 << shift) - 1
    dist = [-1] * n
    dist[s] = 0
    que = [s]
    while que:
        x = heappop(que)
        d = x >> shift
        v = x & mask
        if d > dist[v]:
            continue
        for w, c in g[v]:
            nd = d + c
            dw = dist[w]
            if dw < 0 or nd < dw:
                dist[w] = nd
                heappush(que, nd << shift | w)
    return dist


def dijkstra_prev(
    g: list[list[tuple[int, int]]], s: int
) -> tuple[list[int], list[int]]:
    """dijkstra に加え、最短経路木での各頂点の直前の頂点も返す。

    経路そのものが必要な問題に使い、reconstruct_path で経路に戻す。
    s と到達できない頂点の直前は -1。計算量は dijkstra と同じ。

    Examples:
        >>> g = [[(1, 4), (2, 1)], [], [(1, 2)], []]
        >>> dijkstra_prev(g, 0)
        ([0, 3, 1, -1], [-1, 2, 0, -1])
    """
    n = len(g)
    shift = n.bit_length()
    mask = (1 << shift) - 1
    dist = [-1] * n
    prev = [-1] * n
    dist[s] = 0
    que = [s]
    while que:
        x = heappop(que)
        d = x >> shift
        v = x & mask
        if d > dist[v]:
            continue
        for w, c in g[v]:
            nd = d + c
            dw = dist[w]
            if dw < 0 or nd < dw:
                dist[w] = nd
                prev[w] = v
                heappush(que, nd << shift | w)
    return dist, prev
