import itertools
import random
from collections.abc import Callable
from typing import TypeVar

import pytest

from cplib.graph.bellman_ford import bellman_ford
from cplib.graph.bfs import bfs, bfs_prev
from cplib.graph.bipartite_matching import maximum_bipartite_matching
from cplib.graph.dijkstra import dijkstra, dijkstra_prev
from cplib.graph.doubling import Doubling
from cplib.graph.floyd_warshall import floyd_warshall
from cplib.graph.grid_bfs import grid_bfs
from cplib.graph.lca import LCA
from cplib.graph.lowlink import articulation_points, bridges
from cplib.graph.maxflow import MFGraph
from cplib.graph.mincostflow import MCFGraph
from cplib.graph.minimum_spanning_tree import minimum_spanning_tree
from cplib.graph.reconstruct_path import reconstruct_path
from cplib.graph.rerooting import rerooting
from cplib.graph.scc import SCCGraph
from cplib.graph.topological_sort import topological_sort
from cplib.graph.tree_diameter import tree_diameter
from cplib.graph.two_sat import TwoSAT

T = TypeVar("T")


def reachability(n: int, edges: list[tuple[int, int]]) -> list[list[bool]]:
    reach = [[i == j for j in range(n)] for i in range(n)]
    for u, v in edges:
        reach[u][v] = True
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])
    return reach


@pytest.mark.parametrize("seed", range(30))
def test_scc(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(0, 12)
    edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 3 * n))]
    g = SCCGraph(n)
    for u, v in edges:
        g.add_edge(u, v)
    groups = g.scc()
    reach = reachability(n, edges)

    assert sorted(v for c in groups for v in c) == list(range(n))
    comp = [0] * n
    for i, c in enumerate(groups):
        assert c == sorted(c)
        for v in c:
            comp[v] = i
    for u in range(n):
        for v in range(n):
            assert (comp[u] == comp[v]) == (reach[u][v] and reach[v][u])
    for u, v in edges:
        assert comp[u] <= comp[v]
    num, ids = g.scc_ids()
    assert num == len(groups) and ids == comp


def test_scc_deep_path() -> None:
    n = 10**5
    g = SCCGraph(n)
    for i in range(n - 1):
        g.add_edge(i + 1, i)
    g.add_edge(0, n - 1)
    assert g.scc() == [list(range(n))]


@pytest.mark.parametrize("seed", range(50))
def test_two_sat(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 8)
    clauses = [
        (rng.randrange(n), rng.random() < 0.5, rng.randrange(n), rng.random() < 0.5)
        for _ in range(rng.randint(0, 3 * n))
    ]
    ts = TwoSAT(n)
    for c in clauses:
        ts.add_clause(*c)

    def ok(x: tuple[bool, ...]) -> bool:
        return all(x[i] == f or x[j] == g for i, f, j, g in clauses)

    expected = any(ok(x) for x in itertools.product((False, True), repeat=n))
    assert ts.satisfiable() == expected
    if expected:
        assert ok(tuple(ts.answer()))


@pytest.mark.parametrize("seed", range(20))
def test_floyd_warshall(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 8)
    inf = 10**9
    d = [[0 if i == j else inf for j in range(n)] for i in range(n)]
    for _ in range(rng.randint(0, n * n)):
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            d[u][v] = min(d[u][v], rng.randint(0, 100))

    # 辺を高々 n - 1 本使う経路を全部試す Bellman-Ford 風の愚直解
    expected = [row[:] for row in d]
    for _ in range(n):
        expected = [
            [min(expected[i][k] + d[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)
        ]
    floyd_warshall(d)
    assert d == expected


def random_flow_graph(
    rng: random.Random, n: int, m: int, max_cap: int
) -> list[tuple[int, int, int]]:
    # 自己ループや多重辺も混ぜる
    return [
        (rng.randrange(n), rng.randrange(n), rng.randint(0, max_cap)) for _ in range(m)
    ]


def min_cut_brute(n: int, s: int, t: int, edges: list[tuple[int, int, int]]) -> int:
    # 最大流最小カット定理により、全カットの容量の最小値が最大流
    best = -1
    for bits in range(1 << n):
        if bits >> s & 1 and not bits >> t & 1:
            c = sum(cap for u, v, cap in edges if bits >> u & 1 and not bits >> v & 1)
            if best < 0 or c < best:
                best = c
    return best


def assert_feasible_flow(
    n: int, s: int, t: int, edges: list[tuple[int, int, int]], flows: list[int]
) -> int:
    """流量制約と保存則を確かめ、s から出る正味の流量を返す。"""
    balance = [0] * n
    for (u, v, cap), f in zip(edges, flows, strict=True):
        assert 0 <= f <= cap
        balance[u] -= f
        balance[v] += f
    assert all(balance[v] == 0 for v in range(n) if v not in (s, t))
    assert balance[s] == -balance[t]
    return balance[t]


@pytest.mark.parametrize("seed", range(100))
def test_mf_graph(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(2, 7)
    edges = random_flow_graph(rng, n, rng.randint(0, 3 * n), rng.choice((1, 3, 10)))
    s, t = rng.sample(range(n), 2)
    expected = min_cut_brute(n, s, t, edges)
    limit = rng.choice((None, rng.randint(0, expected + 1)))

    g = MFGraph(n)
    for i, (u, v, cap) in enumerate(edges):
        assert g.add_edge(u, v, cap) == i
    f = g.flow(s, t, limit)
    assert f == (expected if limit is None else min(expected, limit))
    got = g.edges()
    assert [(e.src, e.dst, e.cap) for e in got] == edges
    assert got == [g.get_edge(i) for i in range(len(edges))]
    assert assert_feasible_flow(n, s, t, edges, [e.flow for e in got]) == f

    if limit is None:
        cut = g.min_cut(s)
        assert cut[s] and not cut[t]
        assert sum(c for u, v, c in edges if cut[u] and not cut[v]) == f


@pytest.mark.parametrize("seed", range(30))
def test_mf_graph_incremental(seed: int) -> None:
    # 辺を足しながら flow を何度も呼んだ合計が、全辺での最大流に一致する
    rng = random.Random(seed)
    n = rng.randint(2, 7)
    edges = random_flow_graph(rng, n, rng.randint(0, 3 * n), 5)
    s, t = rng.sample(range(n), 2)
    g = MFGraph(n)
    total = 0
    for u, v, cap in edges:
        g.add_edge(u, v, cap)
        if rng.random() < 0.5:
            total += g.flow(s, t)
    total += g.flow(s, t)
    assert total == min_cut_brute(n, s, t, edges)


def test_mf_graph_change_edge() -> None:
    g = MFGraph(3)
    g.add_edge(0, 1, 5)
    g.add_edge(1, 2, 3)
    assert g.flow(0, 2) == 3
    g.change_edge(1, 10, 0)
    g.change_edge(0, 5, 0)
    assert g.get_edge(1) == MFGraph.Edge(1, 2, 10, 0)
    assert g.flow(0, 2) == 5


def test_mf_graph_unreachable() -> None:
    g = MFGraph(4)
    g.add_edge(0, 1, 3)
    g.add_edge(2, 3, 3)
    g.add_edge(3, 0, 3)
    assert g.flow(0, 3) == 0
    assert g.min_cut(0) == [True, True, False, False]


def test_mf_graph_long_path() -> None:
    n = 10**5
    g = MFGraph(n)
    for i in range(n - 1):
        g.add_edge(i, i + 1, 2)
        g.add_edge(i, i + 1, 1)
    assert g.flow(0, n - 1) == 3


def min_cost_brute(
    n: int, s: int, t: int, edges: list[tuple[int, int, int, int]]
) -> dict[int, int]:
    """流量ごとの最小費用を、全辺の流量を総当たりして求める。"""
    best: dict[int, int] = {}
    caps = [e[2] for e in edges]
    for flows in itertools.product(*(range(c + 1) for c in caps)):
        balance = [0] * n
        for (u, v, _, _), f in zip(edges, flows, strict=True):
            balance[u] -= f
            balance[v] += f
        if any(balance[v] for v in range(n) if v not in (s, t)):
            continue
        if balance[t] < 0:
            continue
        c = sum(e[3] * f for e, f in zip(edges, flows, strict=True))
        if best.get(balance[t], c) >= c:
            best[balance[t]] = c
    return best


@pytest.mark.parametrize("seed", range(100))
def test_mcf_graph(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(2, 5)
    edges = [
        (u, v, c, rng.randint(0, 5))
        for u, v, c in random_flow_graph(rng, n, rng.randint(0, 7), 2)
    ]
    s, t = rng.sample(range(n), 2)
    best = min_cost_brute(n, s, t, edges)
    max_flow = max(best)

    g = MCFGraph(n)
    for i, e in enumerate(edges):
        assert g.add_edge(*e) == i
    points = g.slope(s, t)

    # 折れ線が流量 0..max_flow の最小費用をすべて通り、傾きが狭義単調増加
    assert points[0] == (0, 0) and points[-1] == (max_flow, best[max_flow])
    for (x0, y0), (x1, y1) in itertools.pairwise(points):
        assert x0 < x1
        for x in range(x0, x1 + 1):
            assert best[x] * (x1 - x0) == y0 * (x1 - x) + y1 * (x - x0)
    slopes = [
        (y1 - y0) // (x1 - x0) for (x0, y0), (x1, y1) in itertools.pairwise(points)
    ]
    assert slopes == sorted(set(slopes))

    got = g.edges()
    assert [(e.src, e.dst, e.cap, e.cost) for e in got] == edges
    assert got == [g.get_edge(i) for i in range(len(edges))]
    flows = [e.flow for e in got]
    assert assert_feasible_flow(n, s, t, [e[:3] for e in edges], flows) == max_flow
    assert sum(e.cost * e.flow for e in got) == best[max_flow]


@pytest.mark.parametrize("seed", range(50))
def test_mcf_graph_flow_limit(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(2, 5)
    edges = [
        (u, v, c, rng.randint(0, 5))
        for u, v, c in random_flow_graph(rng, n, rng.randint(0, 7), 2)
    ]
    s, t = rng.sample(range(n), 2)
    best = min_cost_brute(n, s, t, edges)
    limit = rng.randint(0, max(best) + 1)
    g = MCFGraph(n)
    for e in edges:
        g.add_edge(*e)
    f = min(limit, max(best))
    assert g.flow(s, t, limit) == (f, best[f])


def test_mcf_graph_unreachable() -> None:
    g = MCFGraph(3)
    g.add_edge(0, 1, 3, 1)
    g.add_edge(2, 1, 3, 1)
    assert g.slope(0, 2) == [(0, 0)]


def shortest_distances_brute(
    n: int, s: int, edges: list[tuple[int, int, int]]
) -> list[int]:
    dist = [-1] * n
    dist[s] = 0
    for _ in range(n):
        for u, v, c in edges:
            if dist[u] >= 0 and (dist[v] < 0 or dist[u] + c < dist[v]):
                dist[v] = dist[u] + c
    return dist


def assert_shortest_path_tree(
    s: int,
    edges: list[tuple[int, int, int]],
    dist: list[int],
    prev: list[int],
) -> None:
    """prev から復元した経路が、実在する辺をたどって長さ dist になることを確かめる。"""
    cost = {(u, v): c for u, v, c in sorted(edges, key=lambda e: -e[2])}
    assert prev[s] == -1
    for t, d in enumerate(dist):
        if d < 0:
            assert prev[t] == -1
            continue
        path = reconstruct_path(prev, t)
        assert path[0] == s and path[-1] == t
        assert sum(cost[e] for e in itertools.pairwise(path)) == d


def random_weighted_edges(
    rng: random.Random, n: int, max_cost: int
) -> list[tuple[int, int, int]]:
    # 自己ループ・多重辺・到達不能な頂点を含む
    return [
        (rng.randrange(n), rng.randrange(n), rng.randint(0, max_cost))
        for _ in range(rng.randint(0, 3 * n))
    ]


@pytest.mark.parametrize("seed", range(100))
def test_dijkstra(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 10)
    edges = random_weighted_edges(rng, n, rng.choice((0, 3, 10**18)))
    g: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, c in edges:
        g[u].append((v, c))
    s = rng.randrange(n)
    expected = shortest_distances_brute(n, s, edges)
    assert dijkstra(g, s) == expected
    dist, prev = dijkstra_prev(g, s)
    assert dist == expected
    assert_shortest_path_tree(s, edges, dist, prev)


def test_dijkstra_long_path() -> None:
    # 距離が 2^63 を超えて多倍長整数になっても正しい
    n = 10**5
    g: list[list[tuple[int, int]]] = [[(i + 1, 10**18)] for i in range(n - 1)]
    g.append([])
    dist, prev = dijkstra_prev(g, 0)
    assert dist == [i * 10**18 for i in range(n)]
    assert dijkstra(g, 0) == dist
    assert reconstruct_path(prev, n - 1) == list(range(n))


@pytest.mark.parametrize("seed", range(100))
def test_bfs(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 10)
    edges = [(u, v, 1) for u, v, _ in random_weighted_edges(rng, n, 0)]
    g: list[list[int]] = [[] for _ in range(n)]
    for u, v, _ in edges:
        g[u].append(v)
    s = rng.randrange(n)
    expected = shortest_distances_brute(n, s, edges)
    assert bfs(g, s) == expected
    dist, prev = bfs_prev(g, s)
    assert dist == expected
    assert_shortest_path_tree(s, edges, dist, prev)


def test_bfs_long_path() -> None:
    n = 10**5
    g = [[i + 1] for i in range(n - 1)] + [[]]
    dist, prev = bfs_prev(g, 0)
    assert dist == list(range(n)) and bfs(g, 0) == dist
    assert reconstruct_path(prev, n - 1) == list(range(n))


@pytest.mark.parametrize("seed", range(50))
def test_grid_bfs(seed: int) -> None:
    rng = random.Random(seed)
    h, w = rng.randint(1, 6), rng.randint(1, 6)
    wall = rng.choice("#x")
    grid = ["".join(rng.choice((wall, ".", "o")) for _ in range(w)) for _ in range(h)]
    sy, sx = rng.randrange(h), rng.randrange(w)
    grid[sy] = grid[sy][:sx] + "." + grid[sy][sx + 1 :]

    g: list[list[int]] = [[] for _ in range(h * w)]
    for y in range(h):
        for x in range(w):
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] != wall:
                    g[y * w + x].append(ny * w + nx)
    flat = bfs(g, sy * w + sx)
    expected = [flat[y * w : (y + 1) * w] for y in range(h)]
    assert grid_bfs(grid, sy, sx, wall) == expected


def test_grid_bfs_snake() -> None:
    # 2000 x 2000 の一本道の迷路。最も遠いマスまでの手数は通れるマスの数 - 1
    h = w = 2000
    right = "#" * (w - 1) + "."
    left = "." + "#" * (w - 1)
    grid = [
        "." * w if y % 2 == 0 else (right if y % 4 == 1 else left) for y in range(h)
    ]
    dist = grid_bfs(grid, 0, 0)
    assert dist[1][0] == -1
    assert max(map(max, dist)) == sum(row.count(".") for row in grid) - 1


def has_cycle(n: int, edges: list[tuple[int, int]]) -> bool:
    reach = reachability(n, edges)
    return any(u == v or reach[v][u] for u, v in edges)


@pytest.mark.parametrize("seed", range(100))
def test_topological_sort(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(0, 8)
    perm = rng.sample(range(n), n)
    edges: list[tuple[int, int]] = []
    for _ in range(rng.randint(0, 2 * n) if n else 0):
        a, b = sorted((rng.randrange(n), rng.randrange(n)))
        if a != b or rng.random() < 0.1:
            edges.append((perm[a], perm[b]))
    if n and rng.random() < 0.3:
        edges.append((rng.randrange(n), rng.randrange(n)))
    g: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
    order = topological_sort(g)
    if has_cycle(n, edges):
        assert order is None
        return
    assert order is not None and sorted(order) == list(range(n))
    pos = {v: i for i, v in enumerate(order)}
    assert all(pos[u] < pos[v] for u, v in edges)


def test_topological_sort_long_path() -> None:
    n = 10**5
    g = [[i - 1] for i in range(n)]
    g[0] = []
    assert topological_sort(g) == list(range(n - 1, -1, -1))
    g[0] = [n - 1]
    assert topological_sort(g) is None


def random_tree(rng: random.Random, n: int) -> list[list[int]]:
    g: list[list[int]] = [[] for _ in range(n)]
    perm = rng.sample(range(n), n)
    for i in range(1, n):
        p = rng.randrange(max(0, i - 3), i) if rng.random() < 0.5 else rng.randrange(i)
        g[perm[i]].append(perm[p])
        g[perm[p]].append(perm[i])
    return g


@pytest.mark.parametrize("seed", range(50))
def test_lca(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 20)
    g = random_tree(rng, n)
    root = rng.randrange(n)
    parent = [-1] * n
    depth = [0] * n
    order = [root]
    for v in order:
        for w in g[v]:
            if w != parent[v] and w != root:
                parent[w] = v
                depth[w] = depth[v] + 1
                order.append(w)

    def ancestors(v: int) -> list[int]:
        a = [v]
        while a[-1] != root:
            a.append(parent[a[-1]])
        return a

    t = LCA(g, root)
    assert t.depth == depth
    for u in range(n):
        for v in range(n):
            au = set(ancestors(u))
            c = next(x for x in ancestors(v) if x in au)
            assert t.lca(u, v) == c
            assert t.dist(u, v) == depth[u] + depth[v] - 2 * depth[c]


def test_lca_long_path() -> None:
    n = 10**5
    g = [[v for v in (i - 1, i + 1) if 0 <= v < n] for i in range(n)]
    t = LCA(g, n // 2)
    assert t.lca(0, n // 2 - 1) == n // 2 - 1
    assert t.lca(0, n - 1) == n // 2
    assert t.dist(0, n - 1) == n - 1


def component_count(n: int, edges: list[tuple[int, int, int]]) -> int:
    comp = list(range(n))
    for u, v, _ in edges:
        cu, cv = comp[u], comp[v]
        comp = [cu if c == cv else c for c in comp]
    return len(set(comp))


def spanning_forest_cost(n: int, edges: list[tuple[int, int, int]]) -> int:
    """最小全域森のコストを、辺の部分集合の総当たりで求める。"""
    k = n - component_count(n, edges)
    costs = [
        sum(e[2] for e in chosen)
        for chosen in itertools.combinations(edges, k)
        if component_count(n, list(chosen)) == n - k
    ]
    return min(costs)


@pytest.mark.parametrize("seed", range(100))
def test_minimum_spanning_tree(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 6)
    edges = [
        (rng.randrange(n), rng.randrange(n), rng.randint(-5, 5))
        for _ in range(rng.randint(0, 9))
    ]
    total, used = minimum_spanning_tree(n, edges)
    assert total == spanning_forest_cost(n, edges)
    assert total == sum(edges[i][2] for i in used)
    assert len(used) == n - component_count(n, edges)
    assert component_count(n, [edges[i] for i in used]) == n - len(used)
    assert sorted(used, key=lambda i: (edges[i][2], i)) == used


def test_minimum_spanning_tree_disconnected() -> None:
    edges = [(0, 1, 5), (2, 3, 1), (3, 2, 0), (1, 1, -3)]
    assert minimum_spanning_tree(5, edges) == (5, [2, 0])
    assert minimum_spanning_tree(1, []) == (0, [])


def tree_distances(g: list[list[tuple[int, int]]], s: int) -> list[int]:
    dist = [-1] * len(g)
    dist[s] = 0
    stack = [s]
    while stack:
        v = stack.pop()
        for w, c in g[v]:
            if dist[w] < 0:
                dist[w] = dist[v] + c
                stack.append(w)
    return dist


@pytest.mark.parametrize("seed", range(100))
def test_tree_diameter(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 15)
    max_cost = rng.choice((0, 1, 10))
    g: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    cost: dict[tuple[int, int], int] = {}
    for v, adj in enumerate(random_tree(rng, n)):
        for w in adj:
            if v < w:
                c = rng.randint(0, max_cost)
                g[v].append((w, c))
                g[w].append((v, c))
                cost[v, w] = cost[w, v] = c
    expected = max(max(tree_distances(g, s)) for s in range(n))
    length, path = tree_diameter(g)
    assert length == expected
    assert len(set(path)) == len(path)
    assert sum(cost[e] for e in itertools.pairwise(path)) == length


def test_tree_diameter_long_path() -> None:
    n = 10**5
    g: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for i in range(n - 1):
        g[i].append((i + 1, 2))
        g[i + 1].append((i, 2))
    length, path = tree_diameter(g)
    assert length == 2 * (n - 1)
    assert path in (list(range(n)), list(range(n - 1, -1, -1)))
    assert tree_diameter([[]]) == (0, [0])


def bellman_ford_brute(
    n: int, edges: list[tuple[int, int, int]], s: int
) -> tuple[list[int | None], list[bool]]:
    # Floyd-Warshall の値は実在する歩道の長さで、負閉路に触れない対では最短距離になる
    d: list[list[int | None]] = [
        [0 if i == j else None for j in range(n)] for i in range(n)
    ]
    for u, v, w in edges:
        duv = d[u][v]
        if duv is None or w < duv:
            d[u][v] = w
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dik, dkj, dij = d[i][k], d[k][j], d[i][j]
                if dik is not None and dkj is not None:
                    if dij is None or dik + dkj < dij:
                        d[i][j] = dik + dkj
    on_cycle = [(d[c][c] or 0) < 0 for c in range(n)]
    reach = reachability(n, [(u, v) for u, v, _ in edges])
    neg = [
        any(reach[s][c] and on_cycle[c] and reach[c][v] for c in range(n))
        for v in range(n)
    ]
    dist = [None if neg[v] or not reach[s][v] else d[s][v] for v in range(n)]
    return dist, neg


@pytest.mark.parametrize("seed", range(200))
def test_bellman_ford(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 8)
    lo = rng.choice((0, -1, -3, -10))
    edges = [
        (rng.randrange(n), rng.randrange(n), rng.randint(lo, 10))
        for _ in range(rng.randint(0, 3 * n))
    ]
    s = rng.randrange(n)
    assert bellman_ford(n, edges, s) == bellman_ford_brute(n, edges, s)


def test_bellman_ford_long_path() -> None:
    n = 10**5
    edges = [(i, i + 1, -(10**18)) for i in range(n - 1)]
    dist, neg = bellman_ford(n, edges, 0)
    assert dist == [-i * 10**18 for i in range(n)] and not any(neg)
    assert bellman_ford(1, [], 0) == ([0], [False])


def test_bellman_ford_negative_cycle_spreads() -> None:
    # 負閉路は O(nm) 回すべて回るので n を小さくする。閉路の先の長い道に広がる
    n = 3000
    edges = [(i, i + 1, 1) for i in range(n - 1)] + [(2, 1, -2)]
    dist, neg = bellman_ford(n, edges, 0)
    assert dist == [0] + [None] * (n - 1)
    assert neg == [False] + [True] * (n - 1)


def rooted_tree_dp(
    g: list[list[int]],
    root: int,
    merge: Callable[[T, T], T],
    e: T,
    add_edge: Callable[[T, int, int], T],
    add_vertex: Callable[[T, int], T],
) -> T:
    n = len(g)
    parent = [-1] * n
    order = [root]
    for v in order:
        for w in g[v]:
            if w != parent[v]:
                parent[w] = v
                order.append(w)
    acc = [e] * n
    for v in reversed(order[1:]):
        p = parent[v]
        acc[p] = merge(acc[p], add_edge(add_vertex(acc[v], v), v, p))
    return add_vertex(acc[root], root)


def label_vertex(x: frozenset[str], v: int) -> frozenset[str]:
    return frozenset([f"{v}(" + ",".join(sorted(x)) + ")"])


def label_edge(x: frozenset[str], c: int, p: int) -> frozenset[str]:
    return frozenset([f"{c}>{p}[" + ",".join(sorted(x)) + "]"])


def union(a: frozenset[str], b: frozenset[str]) -> frozenset[str]:
    return a | b


@pytest.mark.parametrize("seed", range(50))
def test_rerooting(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 15)
    g = random_tree(rng, n)
    # 根つき木の形と、各コールバックに渡る頂点がすべて文字列に残る
    e: frozenset[str] = frozenset()
    got = rerooting(g, union, e, label_edge, label_vertex)
    assert got == [
        rooted_tree_dp(g, r, union, e, label_edge, label_vertex) for r in range(n)
    ]

    assert rerooting(g, max, 0, lambda x, c, p: x + 1, lambda x, v: x) == [
        max(bfs(g, r)) for r in range(n)
    ]

    def add_size(x: tuple[int, int], v: int) -> tuple[int, int]:
        return x[0] + 1, x[1] + x[0] + 1

    def add_pair(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
        return a[0] + b[0], a[1] + b[1]

    sizes = rerooting(g, add_pair, (0, 0), lambda x, c, p: x, add_size)
    assert sizes == [(n, sum(bfs(g, r)) + n) for r in range(n)]


def test_rerooting_long_path_and_star() -> None:
    n = 10**5
    path = [[v for v in (i - 1, i + 1) if 0 <= v < n] for i in range(n)]
    far = rerooting(path, max, 0, lambda x, c, p: x + 1, lambda x, v: x)
    assert far == [max(i, n - 1 - i) for i in range(n)]
    star = [list(range(1, n))] + [[0] for _ in range(n - 1)]
    far = rerooting(star, max, 0, lambda x, c, p: x + 1, lambda x, v: x)
    assert far == [1] + [2] * (n - 1)


def max_matching_brute(g: list[list[int]]) -> int:
    best = {0: 0}
    for a in g:
        nxt = dict(best)
        for mask, c in best.items():
            for v in a:
                if not mask >> v & 1 and nxt.get(mask | 1 << v, -1) < c + 1:
                    nxt[mask | 1 << v] = c + 1
        best = nxt
    return max(best.values())


def assert_matching(g: list[list[int]], m: int, res: list[int]) -> int:
    assert len(res) == len(g)
    used = [v for v in res if v >= 0]
    assert len(set(used)) == len(used) and all(0 <= v < m for v in used)
    assert all(v < 0 or v in g[u] for u, v in enumerate(res))
    return len(used)


@pytest.mark.parametrize("seed", range(200))
def test_maximum_bipartite_matching(seed: int) -> None:
    rng = random.Random(seed)
    n, m = rng.randint(1, 7), rng.randint(1, 7)
    # 多重辺や、辺のない頂点を含む
    g: list[list[int]] = [[] for _ in range(n)]
    for _ in range(rng.randint(0, 3 * max(n, m))):
        g[rng.randrange(n)].append(rng.randrange(m))
    res = maximum_bipartite_matching(g, m)
    assert assert_matching(g, m, res) == max_matching_brute(g)


def test_maximum_bipartite_matching_empty() -> None:
    assert maximum_bipartite_matching([], 3) == []
    assert maximum_bipartite_matching([[], []], 0) == [-1, -1]


def test_maximum_bipartite_matching_long_path() -> None:
    # 貪欲法で左 i が右 i + 1 と組み、長さ n の増加路が 1 本残る
    n = 10**5
    g = [[i + 1, i] for i in range(n - 1)] + [[n - 1]]
    res = maximum_bipartite_matching(g, n)
    assert assert_matching(g, n, res) == n


def undirected(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    g: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
        g[v].append(u)
    return g


@pytest.mark.parametrize("seed", range(200))
def test_lowlink(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 10)
    # 自己ループ・多重辺・非連結を含む
    edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 2 * n))]
    g = undirected(n, edges)

    def components(es: list[tuple[int, int]]) -> int:
        return component_count(n, [(u, v, 0) for u, v in es])

    base = components(edges)
    expected = [
        (min(u, v), max(u, v))
        for i, (u, v) in enumerate(edges)
        if components(edges[:i] + edges[i + 1 :]) > base
    ]
    got = bridges(g)
    assert sorted(got) == sorted(expected)
    assert all(u < v for u, v in got)

    # v を孤立させると v 自身の分だけ成分が 1 つ増える
    cut = [
        v
        for v in range(n)
        if components([(a, b) for a, b in edges if v not in (a, b)]) - 1 > base
    ]
    assert articulation_points(g) == cut


def test_lowlink_long_path_and_cycle() -> None:
    n = 10**5
    path = undirected(n, [(i, i + 1) for i in range(n - 1)])
    assert sorted(bridges(path)) == [(i, i + 1) for i in range(n - 1)]
    assert articulation_points(path) == list(range(1, n - 1))
    cycle = undirected(n, [(i, (i + 1) % n) for i in range(n)])
    assert bridges(cycle) == [] and articulation_points(cycle) == []
    assert bridges([[]]) == [] and articulation_points([[]]) == []


def jump_brute(nxt: list[int], v: int, k: int) -> int:
    seen: dict[int, int] = {}
    path: list[int] = []
    while v not in seen:
        if len(path) == k:
            return v
        seen[v] = len(path)
        path.append(v)
        v = nxt[v]
    s = seen[v]
    return path[s + (k - s) % (len(path) - s)]


@pytest.mark.parametrize("seed", range(50))
def test_doubling(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 10)
    nxt = [rng.randrange(n) for _ in range(n)]
    max_k = rng.choice((0, 1, rng.randint(2, 100), 10**18))
    d = Doubling(nxt, max_k)
    for _ in range(30):
        v, k = rng.randrange(n), rng.randint(0, max_k)
        assert d.jump(v, k) == jump_brute(nxt, v, k)
    assert d.jump(0, max_k) == jump_brute(nxt, 0, max_k)


def test_doubling_long_tail() -> None:
    # 長さ n / 2 の尻尾の先に長さ n / 2 の閉路
    n = 10**5
    nxt = [i + 1 for i in range(n)]
    nxt[-1] = n // 2
    d = Doubling(nxt, 10**18)
    for k in (0, 1, n - 1, n, 10**18):
        assert d.jump(0, k) == jump_brute(nxt, 0, k)
