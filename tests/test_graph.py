import itertools
import random

import pytest

from cplib.graph.floyd_warshall import floyd_warshall
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
