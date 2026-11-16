def reconstruct_path(prev: list[int], t: int) -> list[int]:
    """直前の頂点の配列 prev から、始点から t までの経路を頂点列で返す。O(経路長)。

    dijkstra_prev や bfs_prev の結果から最短経路を取り出すのに使う。prev は始点で
    -1 になっていること。t が始点から到達できない場合も [t] を返すので、
    到達可能かは距離が -1 でないかで確かめる。

    Examples:
        >>> reconstruct_path([-1, 2, 0, -1], 1)
        [0, 2, 1]
    """
    path = [t]
    while prev[t] >= 0:
        t = prev[t]
        path.append(t)
    path.reverse()
    return path
