def grid_bfs(grid: list[str], sy: int, sx: int, wall: str = "#") -> list[list[int]]:
    """グリッドで、(sy, sx) から上下左右の移動で各マスへ行く最短手数を返す。O(HW)。

    迷路の最短手数を求める問題に使う。文字 wall のマスには入れない。
    到達できないマスと壁のマスは -1。(sy, sx) は壁でないこと。

    Examples:
        >>> grid_bfs(["..#", "#..", "..."], 0, 0)
        [[0, 1, -1], [-1, 2, 3], [4, 3, 4]]
    """
    assert grid[sy][sx] != wall
    h = len(grid)
    w = len(grid[0])
    # 周囲を壁（-2）で囲んだ 1 次元配列にして、境界判定をなくす
    w2 = w + 2
    dist = [-2] * (w2 * (h + 2))
    for y, row in enumerate(grid):
        base = (y + 1) * w2 + 1
        for x, c in enumerate(row):
            if c != wall:
                dist[base + x] = -1
    s = (sy + 1) * w2 + sx + 1
    dist[s] = 0
    que = [s]
    # 4 方向をループにせず展開すると 1〜2 割速い
    for p in que:
        d = dist[p] + 1
        q = p - 1
        if dist[q] == -1:
            dist[q] = d
            que.append(q)
        q = p + 1
        if dist[q] == -1:
            dist[q] = d
            que.append(q)
        q = p - w2
        if dist[q] == -1:
            dist[q] = d
            que.append(q)
        q = p + w2
        if dist[q] == -1:
            dist[q] = d
            que.append(q)
    res = [dist[(y + 1) * w2 + 1 : (y + 1) * w2 + 1 + w] for y in range(h)]
    for row in res:
        for x, d in enumerate(row):
            if d == -2:
                row[x] = -1
    return res
