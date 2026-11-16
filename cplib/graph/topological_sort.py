def topological_sort(g: list[list[int]]) -> list[int] | None:
    """有向グラフの頂点をトポロジカル順に並べて返す。閉路があれば None。O(n + m)。

    DAG 上の DP の計算順を決める問題や、閉路の有無の判定に使う。
    自己ループも閉路とみなす。Kahn 法で、入次数 0 になった順に並べる。

    Examples:
        >>> topological_sort([[1, 2], [], [1]])
        [0, 2, 1]
        >>> topological_sort([[1], [0]]) is None
        True
    """
    n = len(g)
    indeg = [0] * n
    for a in g:
        for w in a:
            indeg[w] += 1
    order = [v for v in range(n) if indeg[v] == 0]
    # 走査中に末尾へ足した頂点もこの for で処理される
    for v in order:
        for w in g[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                order.append(w)
    return order if len(order) == n else None
