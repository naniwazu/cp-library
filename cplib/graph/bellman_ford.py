def bellman_ford(
    n: int, edges: list[tuple[int, int, int]], s: int
) -> tuple[list[int | None], list[bool]]:
    """負のコストを含む有向グラフで、s から各頂点への最短距離を求める。O(nm)。

    負のコストの辺がある最短経路や、負閉路（コストの和が負の閉路）を通っていくらでも
    距離を小さくできるかの判定に使う。edges は (u, v, コスト) の有向辺で、
    自己ループ・多重辺があってもよい。

    (dist, neg) を返す。neg[v] は s から負閉路を経て v に行ける（距離がいくらでも
    小さくなる）か。dist[v] は最短距離で、到達できない頂点と neg[v] が True の頂点は
    None（距離が負になりうるので、到達不能に -1 は使えない）。

    Examples:
        >>> edges = [(0, 1, 4), (0, 2, 5), (2, 1, -3), (3, 4, -1), (4, 3, -1)]
        >>> bellman_ford(5, edges, 0)
        ([0, 2, 5, None, None], [False, False, False, False, False])
        >>> bellman_ford(3, [(0, 1, 1), (1, 1, -1), (1, 2, 1)], 0)
        ([0, None, None], [False, True, True])
    """
    # 実際の距離の絶対値は sum |コスト| 以下なので、これを未到達の印に使える
    inf = sum(abs(w) for _, _, w in edges) + 1
    dist = [inf] * n
    dist[s] = 0
    for _ in range(n - 1):
        changed = False
        for u, v, w in edges:
            du = dist[u]
            if du != inf and du + w < dist[v]:
                dist[v] = du + w
                changed = True
        if not changed:
            break
    # n 回目でも更新される頂点は負閉路の影響を受けており、どの負閉路にも
    # そのような頂点が 1 つはある。そこから到達できる頂点が neg になる
    que: list[int] = []
    for u, v, w in edges:
        du = dist[u]
        if du != inf and du + w < dist[v]:
            dist[v] = du + w
            que.append(v)
    neg = [False] * n
    if que:
        g: list[list[int]] = [[] for _ in range(n)]
        for u, v, _ in edges:
            g[u].append(v)
        for v in que:
            neg[v] = True
        for v in que:
            for w in g[v]:
                if not neg[w]:
                    neg[w] = True
                    que.append(w)
    return [None if neg[v] or dist[v] == inf else dist[v] for v in range(n)], neg
