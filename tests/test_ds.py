import random

import pytest

from cplib.ds.dsu import DSU
from cplib.ds.fenwick_tree import FenwickTree
from cplib.ds.slope_trick import SlopeTrick


@pytest.mark.parametrize("seed", range(20))
def test_fenwick_tree(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(0, 30)
    ft = FenwickTree(n)
    a = [0] * n
    for _ in range(200):
        if n and rng.random() < 0.5:
            p = rng.randrange(n)
            x = rng.randint(-100, 100)
            ft.add(p, x)
            a[p] += x
        else:
            l = rng.randint(0, n)
            r = rng.randint(l, n)
            assert ft.sum(l, r) == sum(a[l:r])


@pytest.mark.parametrize("seed", range(20))
def test_dsu(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(1, 30)
    uf = DSU(n)
    label = list(range(n))
    for _ in range(rng.randint(0, 2 * n)):
        a = rng.randrange(n)
        b = rng.randrange(n)
        x = uf.merge(a, b)
        assert x == uf.leader(a) == uf.leader(b)
        la, lb = label[a], label[b]
        label = [la if c == lb else c for c in label]
        for i in range(n):
            j = rng.randrange(n)
            assert uf.same(i, j) == (label[i] == label[j])
            assert uf.size(i) == label.count(label[i])
        assert uf.group_count() == len(set(label))
    expected: dict[int, list[int]] = {}
    for i, c in enumerate(label):
        expected.setdefault(c, []).append(i)
    assert uf.groups() == sorted(expected.values(), key=lambda g: uf.leader(g[0]))


@pytest.mark.parametrize("seed", range(20))
def test_slope_trick(seed: int) -> None:
    rng = random.Random(seed)
    xs = range(-30, 31)
    f = [0] * len(xs)
    st = SlopeTrick()
    for _ in range(30):
        op = rng.randrange(4)
        a = rng.randint(-20, 20)
        if op == 0:
            st.add_const(a)
            g = [a] * len(xs)
        elif op == 1:
            st.add_x_minus_a(a)
            g = [max(x - a, 0) for x in xs]
        elif op == 2:
            st.add_a_minus_x(a)
            g = [max(a - x, 0) for x in xs]
        else:
            st.add_abs(a)
            g = [abs(x - a) for x in xs]
        f = [u + v for u, v in zip(f, g, strict=True)]
        # 折れ点は [-20, 20] にあるので、最小値はこの範囲で達成される
        assert st.min() == min(f)
