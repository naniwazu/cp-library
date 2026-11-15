def floyd_warshall(d: list[list[int]]) -> None:
    """隣接行列 d を全点対最短距離の行列に破壊的に書き換える。O(n^3)。

    頂点数が数百程度のグラフで、全点対の最短距離が必要な問題に使う。d[i][i] = 0、
    辺がない所は十分大きい値にしておく。負閉路がなければ負辺があってもよい。

    Examples:
        >>> INF = 10**9
        >>> d = [[0, 4, 1], [INF, 0, INF], [INF, 2, 0]]
        >>> floyd_warshall(d)
        >>> d[0]
        [0, 3, 1]
    """
    # 行をローカルに束縛し、dk を enumerate で回すと PyPy で 2 重添字より 2 倍ほど速い
    for k in range(len(d)):
        dk = d[k]
        for di in d:
            dik = di[k]
            for j, x in enumerate(dk):
                x += dik
                if x < di[j]:
                    di[j] = x
