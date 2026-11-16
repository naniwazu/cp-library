from collections.abc import Callable
from typing import TypeVar

_RerootingS = TypeVar("_RerootingS")


def rerooting(
    g: list[list[int]],
    merge: Callable[[_RerootingS, _RerootingS], _RerootingS],
    e: _RerootingS,
    add_edge: Callable[[_RerootingS, int, int], _RerootingS],
    add_vertex: Callable[[_RerootingS, int], _RerootingS],
) -> list[_RerootingS]:
    """木の全頂点について、その頂点を根とした木 DP の値を返す（全方位木 DP）。O(n)。

    各頂点からの最遠距離や距離の総和のように、根を変えながら同じ木 DP を
    繰り返す問題に使う。g は木の隣接リスト（重みなし）。根 r の部分木の値を

        add_vertex(merge(e, add_edge(子 c の部分木の値, c, r), ...), r)

    と定め、各頂点を根とした木全体の値のリストを返す。merge は結合的かつ可換で、
    e はその単位元。add_edge(x, c, p) は子 c から親 p への辺を通す操作で、
    辺の重みはここで (c, p) から引く。各コールバックの呼び出しは O(n) 回。

    Examples:
        各頂点からの最遠距離:

        >>> g = [[1], [0, 2, 3], [1], [1]]
        >>> rerooting(g, max, 0, lambda x, c, p: x + 1, lambda x, v: x)
        [2, 1, 2, 2]

        各頂点を根としたときの部分木サイズの和（距離の総和 + n）。
        値は (部分木の大きさ, 部分木サイズの和):

        >>> res = rerooting(
        ...     g,
        ...     lambda a, b: (a[0] + b[0], a[1] + b[1]),
        ...     (0, 0),
        ...     lambda x, c, p: x,
        ...     lambda x, v: (x[0] + 1, x[1] + x[0] + 1),
        ... )
        >>> [t for _, t in res]
        [9, 7, 9, 9]
    """
    n = len(g)
    parent = [-1] * n
    order = [0]
    for v in order:
        p = parent[v]
        for w in g[v]:
            if w != p:
                parent[w] = v
                order.append(w)
    assert len(order) == n, "g must be a tree"
    # 値は頂点番号でなく BFS 順の位置で持つ。位置 i の頂点の子は位置
    # start[i]..start[i + 1] に連続して並び、メモリを順になめられる
    # （頂点番号で持つより 1.3〜2 倍速い）
    start = [0] * (n + 1)
    pos = 1
    for i in range(n):
        start[i] = pos
        pos += len(g[order[i]]) - (i > 0)
    start[n] = pos
    # up[i]: 位置 i の部分木の値に、親への辺を通したもの
    up = [e] * n
    for i in range(n - 1, 0, -1):
        x = e
        for j in range(start[i], start[i + 1]):
            x = merge(x, up[j])
        v = order[i]
        up[i] = add_edge(add_vertex(x, v), v, parent[v])
    # down[i]: 位置 i の頂点から見た親側の木の値に、位置 i への辺を通したもの
    down = [e] * n
    res = [e] * n
    for i in range(n):
        v = order[i]
        l = start[i]
        r = start[i + 1]
        # 前から: down[j] に、親側と j より前の兄弟を merge したものを仮に置く
        x = down[i]
        for j in range(l, r):
            down[j] = x
            x = merge(x, up[j])
        res[v] = add_vertex(x, v)
        # 後ろから: j より後ろの兄弟を merge しながら down[j] を仕上げる
        if l < r:
            x = up[r - 1]
            down[r - 1] = add_edge(add_vertex(down[r - 1], v), v, order[r - 1])
            for j in range(r - 2, l - 1, -1):
                down[j] = add_edge(add_vertex(merge(down[j], x), v), v, order[j])
                x = merge(x, up[j])
    return res
