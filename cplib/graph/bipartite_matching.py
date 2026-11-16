def maximum_bipartite_matching(g: list[list[int]], m: int) -> list[int]:
    """二部グラフの最大マッチングを求め、左の各頂点と組になる右の頂点を返す。

    仕事の割り当てや、最小点被覆・最大独立集合（König の定理）を求める問題に使う。
    左の頂点が n = len(g) 個、右の頂点が m 個で、g[u] は左の頂点 u とつながる
    右の頂点のリスト（多重辺があってもよい）。組にならない左の頂点は -1。
    scipy.sparse.csgraph.maximum_bipartite_matching(perm_type="column") と同じ形。

    Hopcroft-Karp 法で O(m' sqrt(n + m))（m' は辺数）。MFGraph で解くより 5 倍以上速い。

    Examples:
        >>> g = [[0, 1], [0], [1, 2]]
        >>> res = maximum_bipartite_matching(g, 3)
        >>> res
        [1, 0, 2]
        >>> len(res) - res.count(-1)
        3
    """
    n = len(g)
    match_l = [-1] * n
    match_r = [-1] * m
    for u in range(n):
        for v in g[u]:
            if match_r[v] < 0:
                match_l[u] = v
                match_r[v] = u
                break
    while True:
        # 組になっていない左の頂点から交互路の層を作る
        dist = [-1] * n
        que = [u for u in range(n) if match_l[u] < 0]
        for u in que:
            dist[u] = 0
        found = False
        for u in que:
            d = dist[u] + 1
            for v in g[u]:
                w = match_r[v]
                if w < 0:
                    found = True
                elif dist[w] < 0:
                    dist[w] = d
                    que.append(w)
        if not found:
            break
        # 層に沿った非再帰の DFS で増加路を探す。stack は左の頂点の列で、
        # stack[k + 1] は stack[k] から match_l[stack[k + 1]] を通って来た頂点
        it = [0] * n
        for s in range(n):
            if match_l[s] >= 0:
                continue
            stack = [s]
            while stack:
                u = stack[-1]
                a = g[u]
                i = it[u]
                du = dist[u] + 1
                while i < len(a):
                    v = a[i]
                    i += 1
                    w = match_r[v]
                    if w < 0:
                        for x in reversed(stack):
                            nv = match_l[x]
                            match_l[x] = v
                            match_r[v] = x
                            v = nv
                        stack = []
                        break
                    if dist[w] == du:
                        stack.append(w)
                        break
                else:
                    # この段階では u からもう増加路は見つからない
                    dist[u] = -1
                    stack.pop()
                it[u] = i
    return match_l
