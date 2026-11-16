import itertools
import math
import random
from fractions import Fraction

import pytest

from cplib.misc.angle_sort import argsort_by_angle
from cplib.misc.binary_search import binary_search
from cplib.misc.compress import compress
from cplib.misc.convex_hull import convex_hull
from cplib.misc.cumulative_sum import CumulativeSum2D
from cplib.misc.inversion import inversion_number
from cplib.misc.lis import lis
from cplib.misc.mo import mo_order
from cplib.misc.next_permutation import next_permutation
from cplib.misc.run_length import run_length
from cplib.misc.ternary_search import ternary_search


@pytest.mark.parametrize("seed", range(20))
def test_binary_search(seed: int) -> None:
    rng = random.Random(seed)
    lo = rng.randint(-50, 50)
    hi = lo + rng.randint(1, 50)
    t = rng.randint(lo, hi - 1)
    called: list[int] = []

    def check(x: int) -> bool:
        called.append(x)
        return x <= t

    assert binary_search(lo, hi, check) == t
    assert all(lo < x < hi for x in called)
    assert binary_search(hi, lo, lambda x: x > t) == t + 1


@pytest.mark.parametrize("seed", range(20))
def test_inversion_number(seed: int) -> None:
    rng = random.Random(seed)
    a = [rng.randint(-5, 5) for _ in range(rng.randint(0, 30))]
    expected = sum(x > y for x, y in itertools.combinations(a, 2))
    assert inversion_number(a) == expected


@pytest.mark.parametrize("seed", range(20))
def test_lis(seed: int) -> None:
    rng = random.Random(seed)
    a = [rng.randint(0, 5) for _ in range(rng.randint(0, 10))]
    expected = max(
        (
            len(c)
            for k in range(len(a) + 1)
            for c in itertools.combinations(a, k)
            if all(x < y for x, y in itertools.pairwise(c))
        ),
        default=0,
    )
    assert lis(a) == expected


@pytest.mark.parametrize("seed", range(20))
def test_compress(seed: int) -> None:
    rng = random.Random(seed)
    lim = rng.choice((3, 10**18))
    a = [rng.randint(-lim, lim) for _ in range(rng.choice((0, 1, rng.randint(0, 30))))]
    vals, rank = compress(a)
    assert vals == sorted(set(a))
    assert rank == [sum(v < x for v in set(a)) for x in a]


@pytest.mark.parametrize("seed", range(20))
def test_cumulative_sum_2d(seed: int) -> None:
    rng = random.Random(seed)
    h = rng.choice((0, 1, rng.randint(0, 8)))
    w = rng.choice((0, 1, rng.randint(0, 8)))
    grid = [[rng.randint(-(10**9), 10**9) for _ in range(w)] for _ in range(h)]
    cs = CumulativeSum2D(grid)
    # 行がなければ列数も 0 として扱う
    w = len(grid[0]) if grid else 0
    for y1 in range(h + 1):
        for y2 in range(y1, h + 1):
            for x1 in range(w + 1):
                for x2 in range(x1, w + 1):
                    expected = sum(
                        grid[y][x] for y in range(y1, y2) for x in range(x1, x2)
                    )
                    assert cs.sum(y1, x1, y2, x2) == expected


def test_cumulative_sum_2d_ragged() -> None:
    with pytest.raises(ValueError):
        CumulativeSum2D([[1, 2], [3]])


def naive_run_length(a: list[int]) -> list[tuple[int, int]]:
    res: list[tuple[int, int]] = []
    for x in a:
        if res and res[-1][0] == x:
            res[-1] = (x, res[-1][1] + 1)
        else:
            res.append((x, 1))
    return res


@pytest.mark.parametrize("seed", range(20))
def test_run_length(seed: int) -> None:
    rng = random.Random(seed)
    a = [rng.randint(-1, 1) for _ in range(rng.choice((0, 1, rng.randint(0, 30))))]
    assert run_length(a) == naive_run_length(a)
    assert run_length(iter(a)) == naive_run_length(a)
    s = "".join(rng.choice("ab") for _ in range(rng.randint(0, 30)))
    assert "".join(c * k for c, k in run_length(s)) == s
    assert all(c1 != c2 for (c1, _), (c2, _) in itertools.pairwise(run_length(s)))


@pytest.mark.parametrize("seed", range(20))
def test_mo_order(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.choice((0, 1, rng.randint(0, 50)))
    queries: list[tuple[int, int]] = []
    for _ in range(rng.choice((0, 1, rng.randint(0, 60)))):
        l = rng.randint(0, n)
        queries.append((l, rng.randint(l, n)))
    a = [rng.randint(0, 5) for _ in range(n)]
    order = mo_order(n, queries)
    assert sorted(order) == list(range(len(queries)))
    # 伸縮のループで区間の種類数を求め、愚直解と比べる
    cnt = [0] * 6
    cur = 0
    ans = [0] * len(queries)
    l = r = 0
    for i in order:
        ql, qr = queries[i]
        while l > ql:
            l -= 1
            cur += cnt[a[l]] == 0
            cnt[a[l]] += 1
        while r < qr:
            cur += cnt[a[r]] == 0
            cnt[a[r]] += 1
            r += 1
        while l < ql:
            cnt[a[l]] -= 1
            cur -= cnt[a[l]] == 0
            l += 1
        while r > qr:
            r -= 1
            cnt[a[r]] -= 1
            cur -= cnt[a[r]] == 0
        ans[i] = cur
    assert ans == [len(set(a[l:r])) for l, r in queries]


def test_mo_order_moves() -> None:
    # 伸縮の回数が O(n √q) に収まる
    rng = random.Random(0)
    n = q = 10000
    queries: list[tuple[int, int]] = []
    for _ in range(q):
        l, r = sorted((rng.randint(0, n), rng.randint(0, n)))
        queries.append((l, r))
    l = r = moves = 0
    for i in mo_order(n, queries):
        ql, qr = queries[i]
        moves += abs(l - ql) + abs(r - qr)
        l, r = ql, qr
    assert moves <= 2 * n * math.isqrt(q)


def cross(o: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def naive_convex_hull(points: list[tuple[int, int]]) -> set[tuple[int, int]]:
    # 他の点の凸結合で表せない点（端点）の集合。凸包の頂点は、他の 2 点を結ぶ
    # 線分上になく、他の 3 点の三角形の内部にもない点
    ps = sorted(set(points))
    res: set[tuple[int, int]] = set()
    for p in ps:
        others = [q for q in ps if q != p]
        on_segment = any(
            cross(a, b, p) == 0 and min(a, b) < p < max(a, b)
            for a, b in itertools.combinations(others, 2)
        )
        in_triangle = any(
            (s := [cross(a, b, p), cross(b, c, p), cross(c, a, p)])
            and (all(x >= 0 for x in s) or all(x <= 0 for x in s))
            and cross(a, b, c) != 0
            for a, b, c in itertools.combinations(others, 3)
        )
        if not on_segment and not in_triangle:
            res.add(p)
    return res


@pytest.mark.parametrize("seed", range(40))
def test_convex_hull(seed: int) -> None:
    rng = random.Random(seed)
    lim = rng.choice((0, 1, 2, 5))
    n = rng.choice((0, 1, 2, 3, rng.randint(0, 15)))
    kind = rng.randrange(3)
    points: list[tuple[int, int]] = []
    for _ in range(n):
        x = rng.randint(-lim, lim)
        # 全点が同一直線上になる場合を含める
        y = rng.randint(-lim, lim) if kind == 0 else (2 * x + 1 if kind == 1 else 3)
        points.append((x, y))
    hull = convex_hull(points)
    assert set(hull) == naive_convex_hull(points)
    assert len(hull) == len(set(hull))
    if hull:
        assert hull[0] == min(points)
    m = len(hull)
    if m >= 3:
        # 反時計回りで、同一直線上の 3 点を含まない
        for i in range(m):
            assert cross(hull[i], hull[(i + 1) % m], hull[(i + 2) % m]) > 0


def test_convex_hull_large_coordinates() -> None:
    c = 10**18
    points = [(-c, -c), (c, -c), (c, c), (-c, c), (0, 0), (c, 0)]
    assert convex_hull(points) == [(-c, -c), (c, -c), (c, c), (-c, c)]


def naive_angle(p: tuple[int, int]) -> tuple[int, Fraction]:
    # 象限と、象限内で偏角について単調な有理数
    x, y = p
    if x > 0 and y >= 0:
        return (0, Fraction(y, x))
    if x <= 0 and y > 0:
        return (1, Fraction(-x, y))
    if x < 0 and y <= 0:
        return (2, Fraction(y, x))
    return (3, Fraction(x, -y))


@pytest.mark.parametrize("seed", range(30))
def test_argsort_by_angle(seed: int) -> None:
    rng = random.Random(seed)
    lim = rng.choice((1, 3, 10**9, 10**18))
    points: list[tuple[int, int]] = []
    for _ in range(rng.choice((0, 1, rng.randint(0, 40)))):
        p = (rng.randint(-lim, lim), rng.randint(-lim, lim))
        if rng.random() < 0.3 and points:
            # 同じ点や同じ向きの点
            q = rng.choice(points)
            p = (q[0] * rng.randint(1, 3), q[1] * rng.randint(1, 3)) if lim < 10 else q
        if p != (0, 0):
            points.append(p)
    expected = sorted(range(len(points)), key=lambda i: (naive_angle(points[i]), i))
    assert argsort_by_angle(points) == expected
    if lim == 1:
        # 偏角の誤差が問題にならない小さい座標では atan2 の順とも一致する
        angles = [math.atan2(y, x) % (2 * math.pi) for x, y in points]
        assert [angles[i] for i in expected] == sorted(angles)


def test_argsort_by_angle_close() -> None:
    # 浮動小数点では区別できない近い向き
    c = 10**9
    points = [(c, c - 1), (c - 1, c - 2), (c, c), (-c, -c + 1), (-c + 1, -c + 2)]
    assert argsort_by_angle(points) == [1, 0, 2, 4, 3]


def test_argsort_by_angle_origin() -> None:
    with pytest.raises(AssertionError):
        argsort_by_angle([(1, 0), (0, 0)])


@pytest.mark.parametrize("seed", range(20))
def test_next_permutation(seed: int) -> None:
    rng = random.Random(seed)
    a = sorted(rng.randint(0, 3) for _ in range(rng.choice((0, 1, rng.randint(0, 6)))))
    expected = sorted(set(itertools.permutations(a)))
    got = [tuple(a)]
    while next_permutation(a):
        got.append(tuple(a))
    assert got == expected
    # 最後の順列からは昇順に戻る
    assert a == sorted(a)


def test_next_permutation_middle() -> None:
    a = [1, 3, 2]
    assert next_permutation(a) is True and a == [2, 1, 3]
    s = list("cba")
    assert next_permutation(s) is False and s == ["a", "b", "c"]


@pytest.mark.parametrize("seed", range(30))
def test_ternary_search(seed: int) -> None:
    rng = random.Random(seed)
    lo = rng.randint(-30, 30)
    hi = lo + rng.choice((1, 2, rng.randint(1, 40)))
    # 凸関数: 傾きが広義単調増加な区分線形関数（平らな部分を含みうる）
    slopes = sorted(rng.randint(-3, 3) for _ in range(hi - lo))
    vals = [0]
    for d in slopes:
        vals.append(vals[-1] + d)
    called: list[int] = []

    def f(x: int) -> int:
        called.append(x)
        return vals[x - lo]

    got = ternary_search(lo, hi, f)
    m = min(vals[: hi - lo])
    assert got == lo + vals.index(m)
    assert all(lo <= x < hi for x in called)


def test_ternary_search_float() -> None:
    assert ternary_search(-(10**18), 10**18, lambda x: abs(x - 0.5)) == 0
    assert ternary_search(5, 6, lambda x: -x) == 5
