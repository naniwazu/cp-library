import itertools
import random

import pytest

from cplib.misc.binary_search import binary_search
from cplib.misc.compress import compress
from cplib.misc.cumulative_sum import CumulativeSum2D
from cplib.misc.inversion import inversion_number
from cplib.misc.lis import lis
from cplib.misc.run_length import run_length


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
