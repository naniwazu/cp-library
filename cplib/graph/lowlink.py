def _lowlink(g: list[list[int]]) -> tuple[list[int], list[int], list[int], list[int]]:
    """DFS の行きがけ順、DFS 木の親、行きがけ順の番号 ord、low を返す。"""
    n = len(g)
    ord_ = [-1] * n
    parent = [-1] * n
    order: list[int] = []
    for r in range(n):
        if ord_[r] >= 0:
            continue
        # 隣接頂点をすべて積み、最後に積んだ頂点を親とすると正しい DFS 木になる
        stack = [r]
        while stack:
            v = stack.pop()
            if ord_[v] >= 0:
                continue
            ord_[v] = len(order)
            order.append(v)
            for w in g[v]:
                if ord_[w] < 0:
                    parent[w] = v
                    stack.append(w)
    # 行きがけ順の逆に、隣接頂点の low の最小値をとる。子孫は計算済みで、
    # 祖先はまだ low = ord なので、どちらも low[w] で正しい。親への辺は 1 本だけ除く
    low = ord_[:]
    for v in reversed(order):
        p = parent[v]
        lv = low[v]
        skip = True
        for w in g[v]:
            if w == p and skip:
                skip = False
            elif low[w] < lv:
                lv = low[w]
        low[v] = lv
    return order, parent, ord_, low


def bridges(g: list[list[int]]) -> list[tuple[int, int]]:
    """無向グラフの橋（取り除くと連結成分が増える辺）を返す。O(n + m)。

    ネットワークの単一障害点の列挙や、橋で切った成分ごとに考える問題に使う。
    g は無向グラフの隣接リスト（各辺を両端に入れる）。非連結でもよい。
    多重辺はそれぞれ別の辺として扱い、2 本以上ある辺は橋にならない。自己ループは
    橋にならない。各橋を u < v の (u, v) で 1 回ずつ返す。

    Examples:
        >>> g = [[1, 2], [0, 2], [0, 1, 3], [2, 4, 4], [3, 3]]
        >>> bridges(g)
        [(2, 3)]
    """
    order, parent, ord_, low = _lowlink(g)
    res: list[tuple[int, int]] = []
    for v in order:
        p = parent[v]
        if p >= 0 and low[v] > ord_[p]:
            res.append((p, v) if p < v else (v, p))
    return res


def articulation_points(g: list[list[int]]) -> list[int]:
    """無向グラフの関節点（取り除くと連結成分が増える頂点）を昇順に返す。O(n + m)。

    ネットワークの単一障害点の列挙や、頂点を 1 つ取り除いたときの連結性を
    調べる問題に使う。g は無向グラフの隣接リスト（各辺を両端に入れる）。
    非連結でもよく、多重辺・自己ループがあってもよい。

    Examples:
        >>> g = [[1, 2], [0, 2], [0, 1, 3], [2, 4], [3]]
        >>> articulation_points(g)
        [2, 3]
    """
    order, parent, ord_, low = _lowlink(g)
    cut = [False] * len(g)
    for v in order:
        p = parent[v]
        if p < 0:
            continue
        if parent[p] >= 0:
            if low[v] >= ord_[p]:
                cut[p] = True
        elif ord_[v] > ord_[p] + 1:
            # 根の最初の子は ord が根の次になる。それ以外の子があれば子が 2 つ以上
            cut[p] = True
    return [v for v in range(len(g)) if cut[v]]
