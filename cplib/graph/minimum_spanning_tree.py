def minimum_spanning_tree(
    n: int, edges: list[tuple[int, int, int]]
) -> tuple[int, list[int]]:
    """無向グラフの最小全域木を求め、(総コスト, 使った辺の番号のリスト) を返す。

    頂点をすべて最小コストでつなぐ問題に使う。edges は (u, v, コスト) の無向辺で、
    負のコストや自己ループ・多重辺があってもよい。クラスカル法で O(m log m)。

    非連結なら各連結成分の最小全域木を合わせた森を返す。連結かどうかは
    使った辺の数が n - 1 かで確かめる。辺番号はコストの昇順（同じコストなら
    番号の昇順）に並ぶ。

    Examples:
        >>> edges = [(0, 1, 3), (1, 2, 1), (0, 2, 2), (2, 3, 5)]
        >>> minimum_spanning_tree(4, edges)
        (8, [1, 2, 3])
    """
    m = len(edges)
    # (コスト, 番号) を 1 つの int に詰めてソートすると、key 付きソートより
    # 全体で 3 倍ほど速い
    shift = m.bit_length()
    mask = (1 << shift) - 1
    keys = [edges[i][2] << shift | i for i in range(m)]
    keys.sort()
    # cplib.ds.dsu.DSU のメソッドを呼ぶより 1.5 倍ほど速いので直書きする。
    # 根なら -(成分の大きさ)、そうでなければ親
    parent = [-1] * n
    total = 0
    used: list[int] = []
    for k in keys:
        i = k & mask
        u, v, w = edges[i]
        while parent[u] >= 0:
            if parent[parent[u]] >= 0:
                parent[u] = parent[parent[u]]
            u = parent[u]
        while parent[v] >= 0:
            if parent[parent[v]] >= 0:
                parent[v] = parent[parent[v]]
            v = parent[v]
        if u == v:
            continue
        if parent[u] > parent[v]:
            u, v = v, u
        parent[u] += parent[v]
        parent[v] = u
        total += w
        used.append(i)
    return total, used
