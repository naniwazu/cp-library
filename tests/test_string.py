import random

import pytest

from cplib.string.rolling_hash import RollingHash
from cplib.string.z_algorithm import z_algorithm


@pytest.mark.parametrize("seed", range(20))
def test_rolling_hash(seed: int) -> None:
    rng = random.Random(seed)
    s = "".join(rng.choice("ab") for _ in range(rng.randint(0, 30)))
    a = [ord(c) for c in s]
    t = a[::-1]
    hs = RollingHash(s)
    ht = RollingHash(t)
    n = len(s)
    for _ in range(300):
        k = rng.randint(0, n)
        i = rng.randint(0, n - k)
        j = rng.randint(0, n - k)
        assert (hs.get(i, i + k) == hs.get(j, j + k)) == (s[i : i + k] == s[j : j + k])
        assert (hs.get(i, i + k) == ht.get(j, j + k)) == (a[i : i + k] == t[j : j + k])


def test_rolling_hash_large_values() -> None:
    big = (1 << 61) - 2
    h = RollingHash([big, big, 0, big, big])
    assert h.get(0, 2) == h.get(3, 5)
    assert h.get(0, 2) != h.get(1, 3)


@pytest.mark.parametrize("seed", range(20))
def test_z_algorithm(seed: int) -> None:
    rng = random.Random(seed)
    s = "".join(
        rng.choice("abc"[: rng.randint(1, 3)]) for _ in range(rng.randint(0, 40))
    )
    expected: list[int] = []
    for i in range(len(s)):
        k = 0
        while i + k < len(s) and s[k] == s[i + k]:
            k += 1
        expected.append(k)
    assert z_algorithm(s) == expected
    assert z_algorithm([ord(c) for c in s]) == expected
