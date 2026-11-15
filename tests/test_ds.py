import random
from collections.abc import Callable

import pytest

from cplib.ds.dsu import DSU
from cplib.ds.fenwick_tree import FenwickTree
from cplib.ds.lazy_segtree import LazySegTree
from cplib.ds.segtree import SegTree
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


def _max_right(a: list[str], l: int, f: Callable[[str], bool]) -> int:
    r = l
    while r < len(a) and f("".join(a[l : r + 1])):
        r += 1
    return r


def _min_left(a: list[str], r: int, f: Callable[[str], bool]) -> int:
    l = r
    while l > 0 and f("".join(a[l - 1 : r])):
        l -= 1
    return l


def _count_a_at_most(k: int) -> Callable[[str], bool]:
    # 区間を伸ばすと真から偽にしか変わらない（max_right / min_left の前提）
    return lambda s: s.count("a") <= k


# 2 のべきとその前後、0 と 1 を含める
SIZES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 31, 33]


@pytest.mark.parametrize("n", SIZES)
def test_segtree(n: int) -> None:
    rng = random.Random(n)
    # 非可換な演算（文字列の連結）で、合成の順序まで確かめる
    if rng.random() < 0.5:
        a = [rng.choice("ab") for _ in range(n)]
        st = SegTree(lambda x, y: x + y, "", a.copy())
    else:
        a = [""] * n
        st = SegTree(lambda x, y: x + y, "", n)
    for _ in range(300):
        t = rng.randrange(5)
        if t == 0 and n:
            p = rng.randrange(n)
            a[p] = rng.choice(["a", "b", "ab", ""])
            st.set(p, a[p])
        elif t == 1 and n:
            p = rng.randrange(n)
            assert st.get(p) == a[p]
        elif t == 2:
            l = rng.randint(0, n)
            r = rng.randint(l, n)
            assert st.prod(l, r) == "".join(a[l:r])
        else:
            f = _count_a_at_most(rng.randint(0, 3))
            l = rng.randint(0, n)
            assert st.max_right(l, f) == _max_right(a, l, f)
            r = rng.randint(0, n)
            assert st.min_left(r, f) == _min_left(a, r, f)
        assert st.all_prod() == "".join(a)


@pytest.mark.parametrize("n", SIZES)
def test_lazy_segtree(n: int) -> None:
    rng = random.Random(n)
    # 値は文字列の連結、写像は文字の置き換え（"abc" の並べ替え）で、どちらも非可換
    letters = "abc"

    def mapping(f: tuple[int, ...], s: str) -> str:
        return "".join(letters[f[letters.index(c)]] for c in s)

    def composition(f: tuple[int, ...], g: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(f[x] for x in g)

    def random_map() -> tuple[int, ...]:
        return tuple(rng.sample(range(3), 3))

    v: int | list[str]
    if rng.random() < 0.5:
        a = [rng.choice(letters) for _ in range(n)]
        v = a.copy()
    else:
        a = [""] * n
        v = n
    st = LazySegTree(lambda x, y: x + y, "", mapping, composition, (0, 1, 2), v)
    for _ in range(300):
        t = rng.randrange(6)
        if t == 0 and n:
            p = rng.randrange(n)
            a[p] = rng.choice(letters)
            st.set(p, a[p])
        elif t == 1 and n:
            p = rng.randrange(n)
            assert st.get(p) == a[p]
        elif t == 2:
            l = rng.randint(0, n)
            r = rng.randint(l, n)
            assert st.prod(l, r) == "".join(a[l:r])
        elif t == 3:
            l = rng.randint(0, n)
            r = rng.randint(l, n)
            f = random_map()
            st.apply(l, r, f)
            a[l:r] = [mapping(f, c) for c in a[l:r]]
        else:
            g = _count_a_at_most(rng.randint(0, 3))
            l = rng.randint(0, n)
            assert st.max_right(l, g) == _max_right(a, l, g)
            r = rng.randint(0, n)
            assert st.min_left(r, g) == _min_left(a, r, g)
        assert st.all_prod() == "".join(a)
