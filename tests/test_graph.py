import itertools
import random

import pytest

from cplib.graph.floyd_warshall import floyd_warshall
from cplib.graph.maxflow import MFGraph
from cplib.graph.mincostflow import MCFGraph
from cplib.graph.scc import SCCGraph
from cplib.graph.two_sat import TwoSAT


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
