from typing import NamedTuple


class MFGraph:
    """最大流と最小カットを求める。

    二部マッチング、辺素・点素なパスの最大本数、燃やす埋める（各要素を 2 択に
    振り分けるときの損失の最小化を最小カットに帰着する）などに使う。
    ac-library-python の MFGraph と同じ API。

    Dinic 法で O(n^2 m)。容量がすべて 1 なら O(m min(n^(2/3), m^(1/2)))、
    二部マッチングなら O(m sqrt(n))。実際にはこれよりずっと速いことが多い。

    flow は何度呼んでもよく、前回までの流れに続けて流す（辺を足してから
    流し直す使い方ができる）。

    Examples:
        >>> g = MFGraph(4)
        >>> g.add_edge(0, 1, 2)
        0
        >>> g.add_edge(0, 2, 1)
        1
        >>> g.add_edge(1, 2, 1)
        2
        >>> g.add_edge(1, 3, 1)
        3
        >>> g.add_edge(2, 3, 2)
        4
        >>> g.flow(0, 3)
        3
        >>> g.get_edge(2)
        Edge(src=1, dst=2, cap=1, flow=1)
        >>> g.min_cut(0)
        [True, False, False, False]
    """

    class Edge(NamedTuple):
        src: int
        dst: int
        cap: int
        flow: int

    def __init__(self, n: int) -> None:
        self._n = n
        # 辺 i の順辺が 2i、逆辺が 2i + 1
        self._to: list[int] = []
        self._cap: list[int] = []

    def add_edge(self, src: int, dst: int, cap: int) -> int:
        """容量 cap の辺を足し、その辺番号を返す。"""
        m = len(self._to) >> 1
        self._to.append(dst)
        self._to.append(src)
        self._cap.append(cap)
        self._cap.append(0)
        return m

    def get_edge(self, i: int) -> Edge:
        """辺 i の現在の状態を返す。"""
        return MFGraph.Edge(
            self._to[2 * i + 1],
            self._to[2 * i],
            self._cap[2 * i] + self._cap[2 * i + 1],
            self._cap[2 * i + 1],
        )

    def edges(self) -> list[Edge]:
        """全辺の状態を追加順に返す。"""
        return [self.get_edge(i) for i in range(len(self._to) >> 1)]

    def change_edge(self, i: int, new_cap: int, new_flow: int) -> None:
        """辺 i の容量と流量を書き換える。他の辺は変えない。"""
        self._cap[2 * i] = new_cap - new_flow
        self._cap[2 * i + 1] = new_flow

    def flow(self, s: int, t: int, flow_limit: int | None = None) -> int:
        """s から t へ flow_limit まで流せるだけ流し、流した量を返す。

        flow_limit を省くと流せるだけ流す。
        """
        n = self._n
        to0 = self._to
        cap0 = self._cap
        m2 = len(to0)
        # 辺を始点ごとに CSR 順へ並べ替えて持つ。辺番号経由で引くより速い
        start = [0] * (n + 1)
        for e in range(m2):
            start[to0[e ^ 1] + 1] += 1
        for v in range(n):
            start[v + 1] += start[v]
        pos = start[:]
        where = [0] * m2
        for e in range(m2):
            v = to0[e ^ 1]
            where[e] = pos[v]
            pos[v] += 1
        to = [0] * m2
        cap = [0] * m2
        rev = [0] * m2
        for e in range(m2):
            i = where[e]
            to[i] = to0[e]
            cap[i] = cap0[e]
            rev[i] = where[e ^ 1]
        if flow_limit is None:
            flow_limit = sum(cap[start[s] : start[s + 1]])

        flow = 0
        while flow < flow_limit:
            level = [-1] * n
            level[s] = 0
            que = [s]
            for v in que:
                lv = level[v] + 1
                for i in range(start[v], start[v + 1]):
                    w = to[i]
                    if cap[i] and level[w] < 0:
                        level[w] = lv
                        que.append(w)
                if level[t] >= 0:
                    break
            if level[t] < 0:
                break
            # 非再帰の DFS。path は s から今いる頂点 v までの辺、it[v] は v の次に
            # 調べる辺。流した後は最初に飽和した辺の手前まで戻ってやり直す
            it = start[:]
            path: list[int] = []
            v = s
            while True:
                if v == t:
                    f = flow_limit - flow
                    for i in path:
                        if cap[i] < f:
                            f = cap[i]
                    k = -1
                    for j, i in enumerate(path):
                        cap[i] -= f
                        cap[rev[i]] += f
                        if k < 0 and not cap[i]:
                            k = j
                    flow += f
                    if flow == flow_limit:
                        break
                    v = to[rev[path[k]]]
                    del path[k:]
                    continue
                i = it[v]
                end = start[v + 1]
                lv = level[v] + 1
                while i < end and not (cap[i] and level[to[i]] == lv):
                    i += 1
                it[v] = i
                if i < end:
                    path.append(i)
                    v = to[i]
                else:
                    # v から t へはもう届かないので、この段階では二度と訪れない
                    level[v] = -1
                    if not path:
                        break
                    v = to[rev[path.pop()]]
                    it[v] += 1
        for e in range(m2):
            cap0[e] = cap[where[e]]
        return flow

    def min_cut(self, s: int) -> list[bool]:
        """直前の flow(s, t) の後の残余グラフで s から到達できる頂点を返す。

        True の頂点の集合を S とすると、S から出る辺の容量の和が最小カットで、
        最大流に等しい。O(n + m)。
        """
        n = self._n
        to = self._to
        cap = self._cap
        g: list[list[int]] = [[] for _ in range(n)]
        for e in range(len(to)):
            if cap[e]:
                g[to[e ^ 1]].append(to[e])
        visited = [False] * n
        visited[s] = True
        stack = [s]
        while stack:
            for w in g[stack.pop()]:
                if not visited[w]:
                    visited[w] = True
                    stack.append(w)
        return visited
