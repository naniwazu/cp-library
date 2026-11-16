def _tree_diameter_far(
    g: list[list[tuple[int, int]]], s: int
) -> tuple[list[int], list[int]]:
    """s から各頂点への距離と、s を根としたときの親を返す。"""
    n = len(g)
    dist = [-1] * n
    parent = [-1] * n
    dist[s] = 0
    order = [s]
    for v in order:
        d = dist[v]
        for w, c in g[v]:
            if dist[w] < 0:
                dist[w] = d + c
                parent[w] = v
                order.append(w)
    return dist, parent


def tree_diameter(g: list[list[tuple[int, int]]]) -> tuple[int, list[int]]:
    """重みつきの木の直径を求め、(長さ, 直径をなす経路の頂点列) を返す。O(n)。

    木の最遠点対や、全頂点を巡る最短路（2 × 辺の総和 - 直径）を求める問題に使う。
    g[v] は (行き先, コスト) のリストで、コストは非負。重みなしの木は
    コストをすべて 1 にして渡す。n = 1 なら (0, [0])。

    Examples:
        >>> g = [[(1, 2)], [(0, 2), (2, 1), (3, 5)], [(1, 1)], [(1, 5)]]
        >>> tree_diameter(g)
        (7, [0, 1, 3])
    """
    dist, _ = _tree_diameter_far(g, 0)
    u = dist.index(max(dist))
    dist, parent = _tree_diameter_far(g, u)
    v = dist.index(max(dist))
    path = [v]
    while v != u:
        v = parent[v]
        path.append(v)
    return dist[path[0]], path
