class LCA:
    """根つき木で、2 頂点の最小共通祖先と距離（辺数）に答える。

    木上の 2 点間距離やパス上の頂点判定などに使う。g は木の隣接リスト（重みなし）。
    HL 分解（重軽分解）で、構築 O(n)、クエリ O(log n)。

    Attributes:
        depth: 各頂点の根からの深さ（辺数）。

    Examples:
        >>> g = [[1, 2], [0, 3, 4], [0], [1], [1]]
        >>> t = LCA(g)
        >>> t.lca(3, 4), t.lca(3, 2), t.lca(1, 4)
        (1, 0, 1)
        >>> t.dist(3, 2)
        3
        >>> t.depth
        [0, 1, 1, 2, 2]
    """

    def __init__(self, g: list[list[int]], root: int = 0) -> None:
        n = len(g)
        parent = [-1] * n
        depth = [0] * n
        order = [root]
        for v in order:
            p = parent[v]
            for w in g[v]:
                if w != p:
                    parent[w] = v
                    depth[w] = depth[v] + 1
                    order.append(w)
        assert len(order) == n, "g must be a tree"
        size = [1] * n
        for v in reversed(order):
            if parent[v] >= 0:
                size[parent[v]] += size[v]
        heavy = [-1] * n
        for v in order:
            p = parent[v]
            if p >= 0 and (heavy[p] < 0 or size[v] > size[heavy[p]]):
                heavy[p] = v
        # 重い子を先に訪れる行きがけ順で番号を振ると、各 heavy path が連続した番号になる
        head = [0] * n
        pos = [0] * n
        t = 0
        stack = [root]
        head[root] = root
        while stack:
            v = stack.pop()
            while v >= 0:
                pos[v] = t
                t += 1
                h = heavy[v]
                p = parent[v]
                for w in g[v]:
                    if w != p and w != h:
                        head[w] = w
                        stack.append(w)
                if h >= 0:
                    head[h] = head[v]
                v = h
        self.depth = depth
        self._parent = parent
        self._head = head
        self._pos = pos

    def lca(self, u: int, v: int) -> int:
        """u と v の最小共通祖先を返す。"""
        head = self._head
        pos = self._pos
        parent = self._parent
        while head[u] != head[v]:
            if pos[u] > pos[v]:
                u = parent[head[u]]
            else:
                v = parent[head[v]]
        return u if pos[u] < pos[v] else v

    def dist(self, u: int, v: int) -> int:
        """u と v の間の辺数を返す。"""
        depth = self.depth
        return depth[u] + depth[v] - 2 * depth[self.lca(u, v)]
