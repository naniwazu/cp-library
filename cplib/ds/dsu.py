class DSU:
    """無向グラフに辺を足しながら、連結判定と成分の大きさ取得をならし O(α(n)) で行う。

    辺の追加だけが起こるグラフの連結性や、同値関係のまとめ上げに使う。

    Examples:
        >>> uf = DSU(5)
        >>> uf.merge(0, 1)
        0
        >>> uf.merge(3, 4)
        3
        >>> uf.same(0, 1), uf.same(1, 2)
        (True, False)
        >>> uf.size(1), uf.group_count()
        (2, 3)
        >>> uf.groups()
        [[0, 1], [2], [3, 4]]
    """

    def __init__(self, n: int) -> None:
        self._n = n
        # 代表元なら -(成分の大きさ)、そうでなければ親
        self._parent_or_size = [-1] * n
        self._group_count = n

    def merge(self, a: int, b: int) -> int:
        """a と b を含む成分をつなぎ、新しい代表元を返す。"""
        x = self.leader(a)
        y = self.leader(b)
        if x == y:
            return x
        p = self._parent_or_size
        if p[x] > p[y]:
            x, y = y, x
        p[x] += p[y]
        p[y] = x
        self._group_count -= 1
        return x

    def same(self, a: int, b: int) -> bool:
        """a と b が同じ成分にあるかを返す。"""
        return self.leader(a) == self.leader(b)

    def leader(self, a: int) -> int:
        """a を含む成分の代表元を返す。"""
        p = self._parent_or_size
        r = a
        while p[r] >= 0:
            r = p[r]
        while p[a] >= 0:
            p[a], a = r, p[a]
        return r

    def size(self, a: int) -> int:
        """a を含む成分の頂点数を返す。"""
        return -self._parent_or_size[self.leader(a)]

    def group_count(self) -> int:
        """成分の個数を O(1) で返す。"""
        return self._group_count

    def groups(self) -> list[list[int]]:
        """成分ごとの頂点のリストを返す。O(n α(n))。

        成分は代表元の昇順、成分内の頂点は昇順に並ぶ。
        """
        result: list[list[int]] = [[] for _ in range(self._n)]
        for i in range(self._n):
            result[self.leader(i)].append(i)
        return [g for g in result if g]
