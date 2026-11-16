def bfs(g: list[list[int]], s: int) -> list[int]:
    """重みなしグラフで、s から各頂点への最短距離（辺数）を返す。O(n + m)。

    辺の重みがすべて等しい最短経路や、連結成分の列挙に使う。到達できない頂点の
    距離は -1。

    Examples:
        >>> g = [[1, 2], [3], [3], [], []]
        >>> bfs(g, 0)
        [0, 1, 1, 2, -1]
    """
    dist = [-1] * len(g)
    dist[s] = 0
    que = [s]
    # 走査中に末尾へ足した頂点もこの for で処理される（deque と同程度に速い）
    for v in que:
        d = dist[v] + 1
        for w in g[v]:
            if dist[w] < 0:
                dist[w] = d
                que.append(w)
    return dist


def bfs_prev(g: list[list[int]], s: int) -> tuple[list[int], list[int]]:
    """bfs に加え、最短経路木での各頂点の直前の頂点も返す。

    経路そのものが必要な問題に使い、reconstruct_path で経路に戻す。
    s と到達できない頂点の直前は -1。計算量は bfs と同じ。

    Examples:
        >>> g = [[1, 2], [3], [3], [], []]
        >>> bfs_prev(g, 0)
        ([0, 1, 1, 2, -1], [-1, 0, 0, 1, -1])
    """
    dist = [-1] * len(g)
    prev = [-1] * len(g)
    dist[s] = 0
    que = [s]
    for v in que:
        d = dist[v] + 1
        for w in g[v]:
            if dist[w] < 0:
                dist[w] = d
                prev[w] = v
                que.append(w)
    return dist, prev
