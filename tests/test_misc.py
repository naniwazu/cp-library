import itertools
import random

import pytest

from cplib.misc.binary_search import binary_search
from cplib.misc.inversion import inversion_number
from cplib.misc.lis import lis


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
