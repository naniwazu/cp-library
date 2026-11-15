class SCCGraph:
    """有向グラフを強連結成分に分解する。O(n + m)。

    強連結成分を 1 頂点に縮約した DAG 上で DP する問題や、2-SAT に使う。

    Examples:
        >>> g = SCCGraph(4)
        >>> g.add_edge(0, 1)
        >>> g.add_edge(1, 0)
        >>> g.add_edge(1, 2)
        >>> g.add_edge(3, 2)
        >>> g.scc()
        [[3], [0, 1], [2]]
    """

    def __init__(self, n: int) -> None:
        self._n = n
        self._from: list[int] = []
        self._to: list[int] = []

    def add_edge(self, from_vertex: int, to_vertex: int) -> None:
        """from_vertex から to_vertex への有向辺を足す。"""
        self._from.append(from_vertex)
        self._to.append(to_vertex)

    def scc_ids(self) -> tuple[int, list[int]]:
        """(成分数, 各頂点の成分番号) を返す。成分番号はトポロジカル順に振られる。"""
        n = self._n
        start = [0] * (n + 1)
        for v in self._from:
            start[v + 1] += 1
        for i in range(n):
            start[i + 1] += start[i]
        elist = [0] * len(self._to)
        pos = start[:]
        dst = self._to
        for i, v in enumerate(self._from):
            elist[pos[v]] = dst[i]
            pos[v] += 1

        # 非再帰の Tarjan 法。pos[v] は v の次に調べる辺を指す
        pos = start[:]
        now_ord = group_num = 0
        visited: list[int] = []
        low = [0] * n
        order = [-1] * n
        ids = [0] * n
        for s in range(n):
            if order[s] != -1:
                continue
            low[s] = order[s] = now_ord
            now_ord += 1
            visited.append(s)
            stack = [s]
            while stack:
                v = stack[-1]
                i = pos[v]
                end = start[v + 1]
                while i < end:
                    to = elist[i]
                    i += 1
                    if order[to] == -1:
                        pos[v] = i
                        low[to] = order[to] = now_ord
                        now_ord += 1
                        visited.append(to)
                        stack.append(to)
                        break
                    if order[to] < low[v]:
                        low[v] = order[to]
                else:
                    pos[v] = i
                    stack.pop()
                    if low[v] == order[v]:
                        while True:
                            u = visited.pop()
                            # 確定した頂点は以後 low の更新に影響させない
                            order[u] = n
                            ids[u] = group_num
                            if u == v:
                                break
                        group_num += 1
                    if stack and low[v] < low[stack[-1]]:
                        low[stack[-1]] = low[v]
        # Tarjan 法は逆トポロジカル順に成分を確定させる
        for i in range(n):
            ids[i] = group_num - 1 - ids[i]
        return group_num, ids

    def scc(self) -> list[list[int]]:
        """強連結成分のリストをトポロジカル順に返す。成分内の頂点は昇順。"""
        group_num, ids = self.scc_ids()
        groups: list[list[int]] = [[] for _ in range(group_num)]
        for i, x in enumerate(ids):
            groups[x].append(i)
        return groups
