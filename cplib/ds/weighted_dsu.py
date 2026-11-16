class WeightedDSU:
    """差の制約 p[b] - p[a] = w を足しながら、矛盾の判定と差の取得を行う。

    各頂点に未知の整数 p[i]（ポテンシャル）があり、2 頂点の差の情報が順に与えられる
    問題で使う。矛盾する情報の検出や、つながった 2 頂点の差の計算ができる。
    DSU と同じく merge / same / leader / size を持つ。各操作はならし O(α(n))。

    Examples:
        >>> uf = WeightedDSU(4)
        >>> uf.merge(0, 1, 3)  # p[1] - p[0] = 3
        True
        >>> uf.merge(1, 2, -5)  # p[2] - p[1] = -5
        True
        >>> uf.diff(0, 2), uf.diff(2, 0)  # p[2] - p[0], p[0] - p[2]
        (-2, 2)
        >>> uf.merge(0, 2, 4)  # p[2] - p[0] = -2 と矛盾する
        False
        >>> uf.diff(0, 3) is None, uf.size(2)
        (True, 3)
    """

    def __init__(self, n: int) -> None:
        # 代表元なら -(成分の大きさ)、そうでなければ親
        self._parent_or_size = [-1] * n
        # p[i] - p[親]。代表元では 0
        self._weight = [0] * n

    def merge(self, a: int, b: int, w: int) -> bool:
        """制約 p[b] - p[a] = w を足して True を返す。

        それまでの制約と矛盾するなら何もせず False を返す。
        """
        x = self.leader(a)
        y = self.leader(b)
        wt = self._weight
        # leader で経路を縮約したので wt[a] = p[a] - p[x]、wt[b] = p[b] - p[y]
        w += wt[a] - wt[b]
        if x == y:
            return w == 0
        # ここで w = p[y] - p[x]
        p = self._parent_or_size
        if p[x] > p[y]:
            x, y = y, x
            w = -w
        p[x] += p[y]
        p[y] = x
        wt[y] = w
        return True

    def diff(self, a: int, b: int) -> int | None:
        """p[b] - p[a] を返す。a と b が別の成分にあって決まらないなら None。"""
        if self.leader(a) != self.leader(b):
            return None
        return self._weight[b] - self._weight[a]

    def same(self, a: int, b: int) -> bool:
        """a と b が同じ成分にある（差が決まる）かを返す。"""
        return self.leader(a) == self.leader(b)

    def leader(self, a: int) -> int:
        """a を含む成分の代表元を返す。"""
        p = self._parent_or_size
        wt = self._weight
        r = a
        s = 0
        while p[r] >= 0:
            s += wt[r]
            r = p[r]
        # s = p[a] - p[r]。経路上の頂点を r の直下につなぎ直す
        while p[a] >= 0:
            nxt = p[a]
            w = wt[a]
            p[a] = r
            wt[a] = s
            s -= w
            a = nxt
        return r

    def size(self, a: int) -> int:
        """a を含む成分の頂点数を返す。"""
        return -self._parent_or_size[self.leader(a)]
