from heapq import heappop, heappush
from typing import NamedTuple


class MCFGraph:
    """最小費用流を求める。

    割当問題（n 人に n 個の仕事を 1 対 1 で割り当てて費用和を最小化）、重みつき
    二部マッチング、「k 本の辺素なパスの長さの和の最小」などに使う。容量と費用に
    下限や上限がある問題も、辺を足して最小費用流に帰着できることが多い。
    ac-library-python の MCFGraph と同じ API。

    双対（ポテンシャル）つき Dijkstra で最短路に沿って流す primal-dual 法。
    流量を F として O(F (n + m) log(n + m))。費用は非負であること。

    flow / slope はグラフ 1 つにつき 1 回だけ呼ぶ。呼んだ後の残余グラフには
    費用が負の辺があり、双対を 0 から始める Dijkstra では正しく求まらないため。

    Examples:
        >>> g = MCFGraph(4)
        >>> g.add_edge(0, 1, 2, 1)
        0
        >>> g.add_edge(0, 2, 1, 2)
        1
        >>> g.add_edge(1, 3, 1, 3)
        2
        >>> g.add_edge(2, 3, 2, 1)
        3
        >>> g.add_edge(1, 2, 1, 1)
        4
        >>> g.flow(0, 3)
        (3, 10)
        >>> g.get_edge(4)
        Edge(src=1, dst=2, cap=1, flow=1, cost=1)
    """

    class Edge(NamedTuple):
        src: int
        dst: int
        cap: int
        flow: int
        cost: int

    def __init__(self, n: int) -> None:
        self._n = n
        # 辺 i の順辺が 2i、逆辺が 2i + 1
        self._to: list[int] = []
        self._cap: list[int] = []
        self._cost: list[int] = []

    def add_edge(self, src: int, dst: int, cap: int, cost: int) -> int:
        """容量 cap、単位流量あたりの費用 cost の辺を足し、その辺番号を返す。"""
        m = len(self._to) >> 1
        self._to.append(dst)
        self._to.append(src)
        self._cap.append(cap)
        self._cap.append(0)
        self._cost.append(cost)
        self._cost.append(-cost)
        return m

    def get_edge(self, i: int) -> Edge:
        """辺 i の現在の状態を返す。"""
        return MCFGraph.Edge(
            self._to[2 * i + 1],
            self._to[2 * i],
            self._cap[2 * i] + self._cap[2 * i + 1],
            self._cap[2 * i + 1],
            self._cost[2 * i],
        )

    def edges(self) -> list[Edge]:
        """全辺の状態を追加順に返す。"""
        return [self.get_edge(i) for i in range(len(self._to) >> 1)]

    def flow(self, s: int, t: int, flow_limit: int | None = None) -> tuple[int, int]:
        """s から t へ流量 flow_limit まで流し、(流量, 費用) を返す。

        流量は可能な限り大きく、その中で費用を最小にする。flow_limit を省くと
        流せるだけ流す。
        """
        return self.slope(s, t, flow_limit)[-1]

    def slope(
        self, s: int, t: int, flow_limit: int | None = None
    ) -> list[tuple[int, int]]:
        """流量 x と最小費用 g(x) の関係を表す折れ線の頂点を、流量の昇順に返す。

        g は下に凸な区分線形関数で、先頭は (0, 0)、末尾は flow() の戻り値。
        傾きが同じ点はまとめる。「ちょうど k 流したときの最小費用」を k ごとに
        求めたいときに使う。

        Examples:
            >>> g = MCFGraph(3)
            >>> g.add_edge(0, 1, 2, 1)
            0
            >>> g.add_edge(0, 1, 2, 3)
            1
            >>> g.add_edge(1, 2, 3, 0)
            2
            >>> g.slope(0, 2)
            [(0, 0), (2, 2), (3, 5)]
        """
        n = self._n
        to0 = self._to
        cap0 = self._cap
        cost0 = self._cost
        m2 = len(to0)
        assert all(cost0[e] >= 0 for e in range(0, m2, 2))
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
        cost = [0] * m2
        rev = [0] * m2
        for e in range(m2):
            i = where[e]
            to[i] = to0[e]
            cap[i] = cap0[e]
            cost[i] = cost0[e]
            rev[i] = where[e ^ 1]
        if flow_limit is None:
            flow_limit = sum(cap[start[s] : start[s + 1]])

        # 被約費用は非負なので距離は非負。-1 を未到達に使い、ヒープには
        # 距離 << shift | 頂点 を詰める（タプルを作るより速い）
        shift = n.bit_length()
        mask = (1 << shift) - 1
        dual = [0] * n
        prev = [0] * n
        flow = cost_sum = 0
        prev_cost_per_flow = -1
        result = [(0, 0)]
        while flow < flow_limit:
            dist = [-1] * n
            visited = [False] * n
            dist[s] = 0
            que = [s]
            while que:
                x = heappop(que)
                v = x & mask
                if visited[v]:
                    continue
                visited[v] = True
                if v == t:
                    break
                dv = dual[v] + (x >> shift)
                for i in range(start[v], start[v + 1]):
                    if cap[i]:
                        w = to[i]
                        if visited[w]:
                            continue
                        nd = dv + cost[i] - dual[w]
                        if dist[w] < 0 or nd < dist[w]:
                            dist[w] = nd
                            prev[w] = i
                            heappush(que, nd << shift | w)
            if not visited[t]:
                break
            # 確定した頂点だけ双対を更新すれば、被約費用の非負性が保たれる
            dt = dist[t]
            for v in range(n):
                if visited[v]:
                    dual[v] -= dt - dist[v]
            f = flow_limit - flow
            v = t
            while v != s:
                i = prev[v]
                if cap[i] < f:
                    f = cap[i]
                v = to[rev[i]]
            v = t
            while v != s:
                i = prev[v]
                cap[i] -= f
                cap[rev[i]] += f
                v = to[rev[i]]
            c = -dual[s]
            flow += f
            cost_sum += f * c
            if c == prev_cost_per_flow:
                result.pop()
            result.append((flow, cost_sum))
            prev_cost_per_flow = c
        for e in range(m2):
            cap0[e] = cap[where[e]]
        return result
